"""
Fine-tuning Agent for sagax1
Agent for fine-tuning Hugging Face models on custom datasets
"""

import os
import logging
import tempfile
import json
from typing import Dict, Any, List, Optional, Callable, Union
import pandas as pd

from app.agents.base_agent import BaseAgent
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import Dataset
import torch

class FineTuningAgent(BaseAgent):
    """Agent for fine-tuning models"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the fine-tuning agent
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration dictionary
        """
        super().__init__(agent_id, config)
        
        # Base model configuration with fallback to ensure it's never None
        default_model = "bigscience/bloomz-1b7"  # Good default for fine-tuning
        self.model_id = config.get("model_id") or config.get("model_config", {}).get("model_id", default_model)
        
        # Ensure we have a valid model_id
        if not self.model_id or self.model_id == "None":
            self.model_id = default_model
            self.logger.warning(f"No model_id provided, using default: {default_model}")
        
        self.device = config.get("device", "auto")
        
        # Output configuration
        self.output_dir = config.get("output_dir", "./fine_tuned_models")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # PEFT configuration
        self.lora_r = config.get("lora_r", 16)
        self.lora_alpha = config.get("lora_alpha", 32)
        self.lora_dropout = config.get("lora_dropout", 0.05)
        self.target_modules = config.get("target_modules", None)  # Auto-detect if None
        
        # Training configuration
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.num_train_epochs = config.get("num_train_epochs", 3)
        self.per_device_train_batch_size = config.get("per_device_train_batch_size", 4)
        self.per_device_eval_batch_size = config.get("per_device_eval_batch_size", 4)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_seq_length = config.get("max_seq_length", 512)
        
        # Hugging Face Hub configuration
        self.hub_model_id = config.get("hub_model_id", None)
        self.push_to_hub = config.get("push_to_hub", False)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.is_initialized = False
        
        # Get access to config_manager if available (fix for the error)
        self.config_manager = None
        
        self.logger.info(f"FineTuningAgent {agent_id} initialized with model {self.model_id}")
    
    def _initialize_model_with_auth(self) -> None:
        """Initialize model with proper authentication for gated models"""
        # Check if model requires authentication
        gated_model_providers = ["meta-llama", "mistralai", "google"]
        
        needs_auth = any(provider in self.model_id.lower() for provider in gated_model_providers)
        
        if needs_auth:
            self.logger.info(f"Model {self.model_id} may require authentication")
            
            # Try to get API key from environment
            hf_token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            
            # Try to get from config_manager if available (with safeguard)
            if not hf_token and hasattr(self, 'config_manager') and self.config_manager is not None:
                try:
                    hf_token = self.config_manager.get_hf_api_key()
                except Exception as e:
                    self.logger.warning(f"Error accessing config_manager: {str(e)}")
            
            if not hf_token:
                self.logger.warning(f"No API token found for gated model {self.model_id}")
                self.logger.warning("You may need to set HF_API_TOKEN environment variable or configure API key in settings")
            else:
                self.logger.info("Using API token for model authentication")
        
        # Always use token if available
        token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        
        # Load model with error handling
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                token=token,  # Pass token for gated models
                trust_remote_code=True  # Some models need this
            )
            self.logger.info(f"Model {self.model_id} loaded successfully")
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error loading model: {error_msg}")
            
            # Special handling for common errors
            if "401" in error_msg and "unauthorized" in error_msg.lower():
                raise ValueError(
                    f"Authentication error for model {self.model_id}. Please provide a valid Hugging Face API token "
                    "through the HF_API_TOKEN environment variable or in the application settings. "
                    "You may need to accept the model license on the Hugging Face website."
                )
            elif "404" in error_msg:
                raise ValueError(
                    f"Model {self.model_id} not found. Please check that the model ID is correct. "
                    "Some models require authentication - set your API token in settings."
                )
            else:
                raise




    def initialize(self) -> None:
        """Initialize the model and tokenizer"""
        if self.is_initialized:
            return
        
        try:
            # Load tokenizer
            self.logger.info(f"Loading tokenizer for {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Ensure we have padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_initialized = True
            self.logger.info(f"FineTuningAgent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing fine-tuning agent: {str(e)}")
            raise
    
    def load_instruction_dataset(self, data: List[Dict[str, str]], test_size: float = 0.2) -> Dataset:
        """Load a dataset from instruction/response pairs
        
        Args:
            data: List of dictionaries with instruction, input (optional), and output fields
            test_size: Fraction of data to use for testing
            
        Returns:
            Hugging Face Dataset
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=test_size)
        
        self.logger.info(f"Loaded dataset with {len(dataset['train'])} training and {len(dataset['test'])} test examples")
        self.dataset = dataset
        
        return dataset
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training with robust error handling
        
        Args:
            dataset: Hugging Face Dataset
            
        Returns:
            Preprocessed dataset
        """
        if not self.is_initialized:
            self.initialize()
        
        # Define a function to format text based on model type
        def get_prompt_format(model_id):
            """Get appropriate prompt format for different model families"""
            model_id_lower = model_id.lower()
            
            if any(name in model_id_lower for name in ["llama", "mistral"]):
                return "llama"
            elif "bloom" in model_id_lower:
                return "bloom"
            elif "pythia" in model_id_lower:
                return "pythia"
            elif "opt" in model_id_lower:
                return "opt"
            else:
                return "default"
        
        # Get format for this model
        prompt_format = get_prompt_format(self.model_id)
        self.logger.info(f"Using prompt format: {prompt_format}")
        
        def preprocess_function(examples):
            texts = []
            
            for instruction, input_text, output in zip(
                examples["instruction"], 
                examples.get("input", [""]*len(examples["instruction"])), 
                examples["output"]
            ):
                # Handle missing input field
                input_value = input_text if input_text else ""
                
                # Format text based on model type
                if prompt_format == "llama":
                    if input_value:
                        text = f"<s>[INST] {instruction}\n{input_value} [/INST] {output}</s>"
                    else:
                        text = f"<s>[INST] {instruction} [/INST] {output}</s>"
                elif prompt_format == "bloom":
                    if input_value:
                        text = f"Instruction: {instruction}\nInput: {input_value}\nResponse: {output}"
                    else:
                        text = f"Instruction: {instruction}\nResponse: {output}"
                elif prompt_format == "pythia":
                    # Plain text format
                    if input_value:
                        text = f"Human: {instruction}\n{input_value}\n\nAssistant: {output}"
                    else:
                        text = f"Human: {instruction}\n\nAssistant: {output}"
                elif prompt_format == "opt":
                    if input_value:
                        text = f"User: {instruction}\n{input_value}\nAssistant: {output}"
                    else:
                        text = f"User: {instruction}\nAssistant: {output}"
                else:
                    # Generic default format
                    if input_value:
                        text = f"Instruction: {instruction}\nInput: {input_value}\nResponse: {output}"
                    else:
                        text = f"Instruction: {instruction}\nResponse: {output}"
                    
                texts.append(text)
            
            # Tokenize with safe settings for all models
            tokenized_inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=False  # Avoid token_type_ids issues
            )
            
            # Create clone of input_ids for labels
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            
            return tokenized_inputs
        
        # Apply preprocessing
        try:
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            
            self.logger.info("Dataset preprocessing completed")
            return tokenized_dataset
        except Exception as e:
            self.logger.error(f"Error preprocessing dataset: {str(e)}")
            raise


    # Update the setup_trainer method for better cross-model compatibility
    def setup_trainer(self, tokenized_dataset: Dataset) -> None:
        """Setup trainer for fine-tuning with robust settings
        
        Args:
            tokenized_dataset: Preprocessed dataset
        """
        # Setup training arguments with conservative defaults for compatibility
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=4,  # Reduces memory requirements
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            fp16=True,  # Use mixed precision
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            save_total_limit=2,  # Only keep the 2 most recent checkpoints
            # Conservative settings for cross-model compatibility
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=True,
            optim="adamw_torch",
            gradient_checkpointing=False,  # Disabled for stability
            torch_compile=False,  # Avoid compilation issues
        )
        
        # Create data collator with padding options
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )
        
        self.logger.info("Trainer setup completed")

    
    def load_model(self) -> None:
        """Load the base model for fine-tuning with authentication support"""
        self.logger.info(f"Loading base model {self.model_id}")
        
        # Use specialized method for model initialization
        self._initialize_model_with_auth()

    
    def setup_peft(self) -> None:
        """Setup PEFT for efficient fine-tuning"""
        self.logger.info("Setting up PEFT with LoRA")
        
        # Define target modules by model family
        model_target_modules = {
            # These models use query_key_value
            "bloom": ["query_key_value"],
            
            # These models use q_proj, v_proj, k_proj
            "llama": ["q_proj", "v_proj", "k_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj"],
            "gemma": ["q_proj", "v_proj", "k_proj"],
            "phi": ["q_proj", "v_proj", "k_proj"],
            
            # These models use q_proj, v_proj (without k_proj)
            "gpt_neox": ["query_key_value"],
            "falcon": ["query_key_value"],
            
            # OPT models
            "opt": ["q_proj", "k_proj", "v_proj", "out_proj"],
            
            # Pythia models
            "pythia": ["query_proj", "key_proj", "value_proj"],
            
            # Default fallback
            "default": ["q_proj", "v_proj"]
        }
        
        # Get model type from config or name
        model_type = getattr(self.model.config, "model_type", "").lower()
        model_id_lower = self.model_id.lower()
        
        # Try to match model type or name to known architectures
        target_modules = None
        for family, modules in model_target_modules.items():
            if family in model_type or family in model_id_lower:
                target_modules = modules
                self.logger.info(f"Detected model family: {family}")
                break
        
        # If no match, try examining the model structure
        if not target_modules:
            self.logger.info("Model family not directly recognized, analyzing model structure...")
            
            # Try to find attention modules by examining module names
            attention_module_patterns = {
                "q_proj": ["q_proj", "query_proj", "query"],
                "k_proj": ["k_proj", "key_proj", "key"],
                "v_proj": ["v_proj", "value_proj", "value"]
            }
            
            found_modules = set()
            
            # Examine model structure and find likely target modules
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Check if this is likely an attention component
                    for module_type, patterns in attention_module_patterns.items():
                        if any(pattern in name for pattern in patterns) and name.split('.')[-1] in patterns:
                            found_modules.add(name.split('.')[-1])
                            self.logger.info(f"Found potential attention module: {name}")
            
            if found_modules:
                target_modules = list(found_modules)
                self.logger.info(f"Auto-detected target modules: {target_modules}")
            else:
                # If still no modules found, use default
                target_modules = model_target_modules["default"]
                self.logger.info(f"Could not auto-detect, using default modules: {target_modules}")
        
        self.logger.info(f"Using target modules: {target_modules}")
        
        # Configure LoRA with safeguards
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            
            self.model.print_trainable_parameters()
            self.logger.info("PEFT setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up PEFT: {str(e)}")
            
            # Try with simpler config as fallback
            try:
                self.logger.info("Trying simpler LoRA configuration...")
                
                # Try to find any Linear layers at leaf level
                linear_modules = []
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear) and '.' in name:
                        parts = name.split('.')
                        # Get leaf module name (last part)
                        leaf_name = parts[-1]
                        if leaf_name not in linear_modules:
                            linear_modules.append(leaf_name)
                
                if linear_modules:
                    self.logger.info(f"Found Linear modules: {linear_modules[:10]}...")
                    
                    # Filter to likely attention modules
                    likely_attention = [m for m in linear_modules if any(x in m.lower() for x in ["proj", "query", "key", "value", "attn", "qkv"])]
                    
                    if likely_attention:
                        self.logger.info(f"Using likely attention modules: {likely_attention}")
                        
                        # Retry with these modules
                        lora_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            r=self.lora_r,
                            lora_alpha=self.lora_alpha,
                            lora_dropout=self.lora_dropout,
                            target_modules=likely_attention,
                            bias="none",
                        )
                        
                        self.model = prepare_model_for_kbit_training(self.model)
                        self.model = get_peft_model(self.model, lora_config)
                        
                        self.model.print_trainable_parameters()
                        self.logger.info("PEFT setup completed successfully with fallback configuration")
                        return
                
                # Last resort - try with default configuration
                self.logger.info("Trying with default LoRA configuration...")
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,  # Reduced rank for stability
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["query", "value"],  # Very generic names
                    bias="none",
                )
                
                self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, lora_config)
                
                self.model.print_trainable_parameters()
                self.logger.info("PEFT setup completed with minimal configuration")
                
            except Exception as fallback_error:
                self.logger.error(f"All PEFT configurations failed: {str(fallback_error)}")
                raise RuntimeError(f"Could not configure PEFT for model {self.model_id}. Please try a different model.")

    
   
    
    def train(self, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Fine-tune the model with comprehensive error handling
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training results
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        self.logger.info("Starting fine-tuning")
        
        # Custom callback to report progress
        if progress_callback:
            from transformers.trainer_callback import TrainerCallback
            
            class ProgressReportCallback(TrainerCallback):
                def __init__(self, progress_fn):
                    self.progress_fn = progress_fn
                    self.last_log = {}
                    self.current_epoch = 0
                    self.total_epochs = 0
                    
                def on_train_begin(self, args, state, control, **kwargs):
                    self.total_epochs = args.num_train_epochs
                    self.progress_fn(f"Training started. Total epochs: {self.total_epochs}")
                    
                def on_epoch_begin(self, args, state, control, **kwargs):
                    self.current_epoch = state.epoch
                    self.progress_fn(f"Starting epoch {self.current_epoch+1}/{self.total_epochs}")
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None:
                        return
                    
                    if logs != self.last_log:
                        # Extract useful metrics
                        step = logs.get("step", 0)
                        loss = logs.get("loss", "N/A")
                        epoch = logs.get("epoch", 0)
                        
                        # Calculate approximate progress percentage
                        if self.total_epochs > 0:
                            progress_pct = min(int((epoch / self.total_epochs) * 100), 99)
                        else:
                            progress_pct = 0
                            
                        progress_msg = f"Training: Step={step}, Loss={loss}, Epoch={epoch:.2f} ({progress_pct}% complete)"
                        self.progress_fn(progress_msg)
                        self.last_log = logs.copy()
                
                def on_train_end(self, args, state, control, **kwargs):
                    self.progress_fn("Training completed!")
            
            self.trainer.add_callback(ProgressReportCallback(progress_callback))
        
        # Start training with comprehensive error handling
        try:
            train_result = self.trainer.train()
            
            self.logger.info("Fine-tuning completed")
            
            # Save model and tokenizer
            try:
                self.logger.info("Saving fine-tuned model")
                self.trainer.save_model()
                self.tokenizer.save_pretrained(self.output_dir)
            except Exception as save_error:
                self.logger.error(f"Error saving model: {str(save_error)}")
                # Continue despite save error, we still have metrics
            
            # Report metrics
            metrics = train_result.metrics
            try:
                self.trainer.log_metrics("train", metrics)
                self.trainer.save_metrics("train", metrics)
            except Exception as metrics_error:
                self.logger.error(f"Error saving metrics: {str(metrics_error)}")
                # Continue despite metrics error
            
            if self.push_to_hub:
                try:
                    self.logger.info(f"Pushing model to Hugging Face Hub: {self.hub_model_id}")
                    self.trainer.push_to_hub()
                except Exception as hub_error:
                    self.logger.error(f"Error pushing to Hub: {str(hub_error)}")
                    # Continue despite hub error
            
            return metrics
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error during fine-tuning: {error_str}")
            
            # Create a structured error response
            error_response = {
                "status": "error",
                "message": error_str,
                "type": type(e).__name__
            }
            
            # Add diagnostic information
            try:
                error_response["model_id"] = self.model_id
                error_response["device"] = self.device
                error_response["output_dir"] = self.output_dir
                
                # Get GPU info if available
                try:
                    import torch
                    error_response["cuda_available"] = torch.cuda.is_available()
                    if torch.cuda.is_available():
                        error_response["cuda_device_count"] = torch.cuda.device_count()
                        error_response["cuda_device_name"] = torch.cuda.get_device_name(0)
                        error_response["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory
                except Exception:
                    pass
                    
                # Memory usage info
                try:
                    import psutil
                    error_response["total_memory"] = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
                    error_response["available_memory"] = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                except Exception:
                    pass
            except Exception:
                pass
                
            # Add error classification for better UI feedback
            error_type = "unknown"
            if "memory" in error_str.lower() or "cuda out of memory" in error_str.lower():
                error_type = "memory"
            elif "authentication" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
                error_type = "authentication"
            elif "not found" in error_str.lower() or "404" in error_str:
                error_type = "not_found"
            elif "target modules" in error_str.lower():
                error_type = "model_compatibility"
            elif "index out of range" in error_str.lower():
                error_type = "data_processing"
            
            error_response["error_type"] = error_type
            
            raise RuntimeError(json.dumps(error_response))
    
    def generate_sample_response(self, instruction: str, input_text: str = "") -> str:
        """Generate a sample response from the fine-tuned model
        
        Args:
            instruction: Instruction text
            input_text: Optional input text
            
        Returns:
            Generated response
        """
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Fine-tuned model not found at {self.output_dir}")
        
        # Load fine-tuned model
        self.logger.info(f"Loading fine-tuned model from {self.output_dir}")
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(
                base_model,
                self.output_dir,
                torch_dtype=torch.float16
            )
            
            # Merge adapter weights with the base model for better performance
            model = model.merge_and_unload()
            
            # Format the input
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
            else:
                prompt = f"Instruction: {instruction}\nResponse:"
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate a response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the model's response (after our prompt)
            response = full_response[len(prompt):]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input
        
        Args:
            input_text: Input text for the agent
            callback: Optional callback for streaming responses
            
        Returns:
            Agent output text
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Parse input as JSON
            try:
                command = json.loads(input_text)
                action = command.get("action", "")
                
                if action == "fine_tune":
                    # Fine-tune a model
                    dataset = command.get("dataset", [])
                    test_size = command.get("test_size", 0.2)
                    custom_config = command.get("config", {})
                    
                    # Update config with custom values
                    for key, value in custom_config.items():
                        setattr(self, key, value)
                    
                    # Load dataset
                    self.load_instruction_dataset(dataset, test_size)
                    
                    # Preprocess dataset
                    tokenized_dataset = self.preprocess_dataset(self.dataset)
                    
                    # Load model
                    self.load_model()
                    
                    # Setup PEFT
                    self.setup_peft()
                    
                    # Setup trainer
                    self.setup_trainer(tokenized_dataset)
                    
                    # Train model
                    metrics = self.train(progress_callback=callback)
                    
                    # Generate response
                    result = {
                        "status": "success",
                        "message": "Fine-tuning completed successfully",
                        "metrics": metrics,
                        "model_path": self.output_dir
                    }
                    
                    if self.push_to_hub:
                        result["hub_url"] = f"https://huggingface.co/{self.hub_model_id}"
                    
                    return json.dumps(result, indent=2)
                
                elif action == "generate":
                    # Generate a response using the fine-tuned model
                    instruction = command.get("instruction", "")
                    input_value = command.get("input", "")
                    
                    response = self.generate_sample_response(instruction, input_value)
                    
                    return response
                
                else:
                    return f"Unknown action: {action}. Supported actions are 'fine_tune' and 'generate'."
            
            except json.JSONDecodeError:
                # Not valid JSON, treat as a prompt for generation
                return self.generate_sample_response(input_text)
                
        except Exception as e:
            error_msg = f"Error in fine-tuning agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has
        
        Returns:
            List of capability names
        """
        return [
            "fine_tuning",
            "model_training",
            "low_rank_adaptation",
            "text_generation"
        ]
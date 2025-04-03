"""
Fine-tuning Agent for SagaX1
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
                model_id: Hugging Face model ID to fine-tune
                device: Device to use (e.g., 'cpu', 'cuda', 'mps')
                output_dir: Directory to save the fine-tuned model
                lora_r: LoRA rank
                lora_alpha: LoRA alpha
                lora_dropout: LoRA dropout
                learning_rate: Learning rate for fine-tuning
                num_train_epochs: Number of training epochs
                per_device_train_batch_size: Batch size for training
                per_device_eval_batch_size: Batch size for evaluation
                warmup_steps: Number of warmup steps
                weight_decay: Weight decay
                hub_model_id: Hugging Face Hub model ID for uploading
                push_to_hub: Whether to push to Hugging Face Hub
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
        
        self.logger.info(f"FineTuningAgent {agent_id} initialized with model {self.model_id}")
    
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
        """Preprocess the dataset for training
        
        Args:
            dataset: Hugging Face Dataset
            
        Returns:
            Preprocessed dataset
        """
        if not self.is_initialized:
            self.initialize()
        
        def preprocess_function(examples):
            texts = []
            for instruction, input_text, output in zip(
                examples["instruction"], 
                examples.get("input", [""]*len(examples["instruction"])), 
                examples["output"]
            ):
                # Handle missing input field
                input_value = input_text if input_text else ""
                
                if input_value:
                    text = f"Instruction: {instruction}\nInput: {input_value}\nResponse: {output}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {output}"
                texts.append(text)
            
            # IMPORTANT: Don't include token_type_ids in the output - this causes issues with some models
            tokenized_inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=False  # This is the key fix
            )
            
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

    # Also update the setup_trainer method to add additional fixes for common issues

    def setup_trainer(self, tokenized_dataset: Dataset) -> None:
        """Setup trainer for fine-tuning
        
        Args:
            tokenized_dataset: Preprocessed dataset
        """
        # Remove any problematic columns from the dataset
        columns_to_remove = ["token_type_ids"]
        for split in tokenized_dataset:
            for column in columns_to_remove:
                if column in tokenized_dataset[split].column_names:
                    tokenized_dataset[split] = tokenized_dataset[split].remove_columns(column)
        
        # Setup training arguments
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
            # Added settings for better compatbility
            remove_unused_columns=True,  # Important fix for some models
            gradient_checkpointing=False,  # Disable for smaller models
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
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
        """Load the base model for fine-tuning"""
        self.logger.info(f"Loading base model {self.model_id}")
        
        try:
            # Load base model with float16 precision
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            
            self.logger.info(f"Model {self.model_id} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_peft(self) -> None:
        """Setup PEFT for efficient fine-tuning"""
        self.logger.info("Setting up PEFT with LoRA")
        
        # Detect target modules if not specified
        if not self.target_modules:
            # Get the model's architecture from its config
            model_arch = getattr(self.model.config, "model_type", "").lower()
            self.logger.info(f"Detected model architecture: {model_arch}")
            
            # Different models use different layer names
            if "llama" in model_arch or "llama" in self.model_id.lower():
                self.target_modules = ["q_proj", "v_proj"]
            elif "bloom" in model_arch or "bloom" in self.model_id.lower():
                self.target_modules = ["query_key_value"]
            elif "gpt" in model_arch or "gpt" in self.model_id.lower():
                self.target_modules = ["c_attn"]
            elif "mistral" in model_arch or "mistral" in self.model_id.lower():
                self.target_modules = ["q_proj", "v_proj"]
            elif "smollm" in model_arch or "smollm" in self.model_id.lower():
                self.target_modules = ["q_proj", "v_proj", "k_proj"]
            else:
                # Try to auto-detect by checking for common attention names
                found_modules = []
                for name, _ in self.model.named_modules():
                    if any(module_name in name for module_name in ["q_proj", "query", "attention"]):
                        found_modules.append(name)
                
                # If found modules, use those that end with known attention names
                if found_modules:
                    module_candidates = []
                    for module in found_modules:
                        parts = module.split('.')
                        if parts[-1] in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "query_key_value"]:
                            module_candidates.append(parts[-1])
                    
                    if module_candidates:
                        self.target_modules = list(set(module_candidates))
                        self.logger.info(f"Auto-detected target modules: {self.target_modules}")
                    else:
                        # Fallback to most common options if we couldn't determine specifics
                        self.target_modules = ["q_proj", "v_proj"]
                else:
                    # Fallback for unknown architectures
                    self.target_modules = ["q_proj", "v_proj"]
        
        self.logger.info(f"Using target modules: {self.target_modules}")
        
        # Configure LoRA
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )
            
            # Prepare model for PEFT
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.logger.info("PEFT setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up PEFT: {str(e)}")
            
            # Try one more approach with a more general module finder
            if "not found in the base model" in str(e):
                try:
                    self.logger.info("Trying alternative target module detection...")
                    
                    # Extract all module names and find potential attention modules
                    all_modules = [name for name, _ in self.model.named_modules()]
                    attention_candidates = []
                    
                    # Keywords that might indicate attention-related modules
                    attention_keywords = ["attention", "attn", "self", "proj", "query", "key", "value"]
                    
                    for module in all_modules:
                        if any(keyword in module.lower() for keyword in attention_keywords):
                            parts = module.split(".")
                            if len(parts) > 1:  # We want leaf modules, not parent containers
                                attention_candidates.append(module)
                    
                    if attention_candidates:
                        self.logger.info(f"Found potential attention modules: {attention_candidates[:5]}...")
                        
                        # Get the module names at the correct level
                        possible_targets = []
                        for module in attention_candidates:
                            parts = module.split(".")
                            # Get the last part and check if it's likely to be a target
                            if len(parts) > 1 and not parts[-1].isdigit():
                                possible_targets.append(parts[-1])
                        
                        # Remove duplicates
                        unique_targets = list(set(possible_targets))
                        
                        if unique_targets:
                            self.logger.info(f"Using detected target modules: {unique_targets}")
                            
                            # Try again with the detected modules
                            lora_config = LoraConfig(
                                task_type=TaskType.CAUSAL_LM,
                                r=self.lora_r,
                                lora_alpha=self.lora_alpha,
                                lora_dropout=self.lora_dropout,
                                target_modules=unique_targets,
                                bias="none",
                            )
                            
                            self.model = prepare_model_for_kbit_training(self.model)
                            self.model = get_peft_model(self.model, lora_config)
                            
                            # Print trainable parameters
                            self.model.print_trainable_parameters()
                            
                            self.logger.info("PEFT setup completed successfully with alternative modules")
                            return
                    
                    # If we get here, the fallback approach didn't work either
                    raise e
                except Exception as fallback_error:
                    self.logger.error(f"Fallback approach also failed: {str(fallback_error)}")
                    raise
    
   
    
    def train(self, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Fine-tune the model
        
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
            # Use the standard Transformers callback class instead of creating a custom one
            from transformers.trainer_callback import TrainerCallback
            
            class ProgressReportCallback(TrainerCallback):
                def __init__(self, progress_fn):
                    self.progress_fn = progress_fn
                    self.last_log = {}
                    self.step = 0
                    
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None:
                        return
                        
                    if logs != self.last_log:
                        # Report progress
                        self.step += 1
                        loss = logs.get("loss", "N/A")
                        epoch = logs.get("epoch", 0)
                        progress_msg = f"Training: Step={self.step}, Loss={loss}, Epoch={epoch:.2f}"
                        self.progress_fn(progress_msg)
                        self.last_log = logs.copy()
            
            self.trainer.add_callback(ProgressReportCallback(progress_callback))
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            self.logger.info("Fine-tuning completed")
            
            # Save model and tokenizer
            self.logger.info("Saving fine-tuned model")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Report metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            if self.push_to_hub:
                self.logger.info(f"Pushing model to Hugging Face Hub: {self.hub_model_id}")
                self.trainer.push_to_hub()
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
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
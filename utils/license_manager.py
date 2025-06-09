"""
License Manager for sagax1
Handles license validation with SendOwl and local license storage
"""

import os
import json
import logging
import hashlib
import requests
import platform
import uuid
from datetime import datetime
from pathlib import Path

class LicenseManager:
    """Manages license validation and storage for sagax1"""
    
    def __init__(self, config_path=None):
        """Initialize the license manager
        
        Args:
            config_path: Path to config directory, defaults to ~/.sagax1
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up config path
        if config_path is None:
            self.config_path = os.path.join(Path.home(), ".sagax1")
        else:
            self.config_path = config_path
            
        os.makedirs(self.config_path, exist_ok=True)
        
        # License file path
        self.license_file = os.path.join(self.config_path, "license.json")
        
        # SendOwl API configuration (should be configured in actual deployment)
        self.sendowl_api_url = "https://www.sendowl.com/api/v1/licenses/valid"
        
        # Generate a machine ID (used for license binding)
        self.machine_id = self._get_machine_id()
    
    def _get_machine_id(self):
        """Generate a unique machine ID based on hardware information
        
        Returns:
            A unique identifier for this machine
        """
        # Collect system information that doesn't change often
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node()
        }
        
        # Try to get more persistent hardware identifiers
        try:
            if platform.system() == "Windows":
                # On Windows, use the MachineGuid from registry
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                    r"SOFTWARE\Microsoft\Cryptography") as key:
                    system_info["guid"] = winreg.QueryValueEx(key, "MachineGuid")[0]
            elif platform.system() == "Darwin":  # macOS
                # On macOS, use IOPlatformUUID
                import subprocess
                system_info["uuid"] = subprocess.check_output(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
                ).decode().split("IOPlatformUUID")[1].split("\"")[2]
            else:  # Linux and others
                # Try to use the machine-id
                machine_id_paths = [
                    "/etc/machine-id",
                    "/var/lib/dbus/machine-id"
                ]
                for path in machine_id_paths:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            system_info["machine_id"] = f.read().strip()
                        break
        except Exception as e:
            self.logger.warning(f"Could not get detailed machine ID: {e}")
        
        # Fallback if we couldn't get specific IDs
        if not any(key in system_info for key in ["guid", "uuid", "machine_id"]):
            system_info["fallback_id"] = str(uuid.getnode())  # MAC address-based
        
        # Generate a hash based on the collected information
        system_str = json.dumps(system_info, sort_keys=True)
        return hashlib.sha256(system_str.encode()).hexdigest()
    
    def is_licensed(self):
        """Check if the software is licensed
        
        Returns:
            True if licensed, False otherwise
        """
        # First check for local license file
        if os.path.exists(self.license_file):
            try:
                with open(self.license_file, "r") as f:
                    license_data = json.load(f)
                
                # Verify the license is valid and bound to this machine
                if license_data.get("machine_id") == self.machine_id:
                    # Check if the license has expired
                    expires_date = license_data.get("expires")
                    if expires_date and expires_date != "never":
                        try:
                            expires = datetime.fromisoformat(expires_date)
                            if expires < datetime.now():
                                self.logger.warning("License has expired")
                                return False
                        except ValueError:
                            # If we can't parse the date, assume it's valid
                            pass
                    
                    self.logger.info("Valid license found")
                    return True
                else:
                    self.logger.warning("License is bound to another machine")
            except Exception as e:
                self.logger.error(f"Error reading license file: {e}")
        
        self.logger.info("No valid license found")
        return False
    
    def validate_license(self, license_key):
        """Validate a license key with SendOwl
        
        Args:
            license_key: License key to validate
            
        Returns:
            (bool, str): Success status and message
        """
        # Skip online validation if we already have a valid local license
        if self.is_licensed():
            return True, "License already validated"
        
        try:
            # Call SendOwl API to validate the license
            # Note: In a real implementation, you would use your SendOwl API credentials
            # This is a simplified example
            response = self._validate_with_sendowl(license_key)
            
            if response.get("valid", False):
                # Save the license information locally
                license_data = {
                    "key": license_key,
                    "machine_id": self.machine_id,
                    "name": response.get("customer_name", "User"),
                    "email": response.get("customer_email", ""),
                    "issued": datetime.now().isoformat(),
                    "expires": response.get("expires", "never"),
                    "product": "sagax1"
                }
                
                self._save_license(license_data)
                return True, "License validated successfully"
            else:
                return False, response.get("message", "Invalid license key")
                
        except Exception as e:
            self.logger.error(f"Error validating license: {e}")
            return False, f"Error validating license: {str(e)}"
    
    def _validate_with_sendowl(self, license_key):
        """Validate license with SendOwl API - this is a simplified example
        
        In a real implementation, you would make an actual API call to SendOwl.
        This example simulates a successful validation.
        
        Args:
            license_key: License key to validate
            
        Returns:
            Response data
        """
        # In a real implementation, you would make a request to SendOwl API
        # For this example, we'll simply accept any license key that is 16+ characters
        if len(license_key) >= 16:
            # Simulate a successful response
            return {
                "valid": True,
                "license_key": license_key,
                "customer_name": "Test User",
                "customer_email": "user@example.com",
                "expires": "never"
            }
        else:
            # Simulate an error response
            return {
                "valid": False,
                "message": "Invalid license key"
            }
            
        # Real SendOwl implementation :
        """
        response = requests.get(
            self.sendowl_api_url,
            params={"license_key": license_key},
            auth=("sendowl_api_key", "sendowl_api_secret")
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"valid": False, "message": f"API error: {response.status_code}"}
        """
    
    def _save_license(self, license_data):
        """Save license data to local file
        
        Args:
            license_data: License data dictionary
        """
        try:
            with open(self.license_file, "w") as f:
                json.dump(license_data, f, indent=2)
            
            self.logger.info("License saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving license: {e}")
# =============================================================================
# EKS Module - Required Providers
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    
    # Required for OIDC certificate thumbprint
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0"
    }
  }
}

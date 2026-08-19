terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "this" {
  location      = var.region
  repository_id = "advisor-match-service"
  format        = "DOCKER"
  description   = "Container images for the advisor matching API"
}

resource "google_cloud_run_v2_service" "this" {
  name     = "advisor-match-service"
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.this.repository_id}/advisor-match-service:latest"

      env {
        name  = "ELASTICSEARCH_URL"
        value = var.elasticsearch_url
      }

      resources {
        limits = {
          # PyTorch + the embedding model need more headroom than a typical
          # Cloud Run default (256Mi/1 vCPU) would give.
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

# Client-advisor matching is meant to be publicly reachable; a real deployment
# would put this behind an API gateway or IAP rather than allUsers directly.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.this.name
  location = google_cloud_run_v2_service.this.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

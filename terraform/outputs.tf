output "cloud_run_url" {
  value = google_cloud_run_v2_service.this.uri
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.this.name
}

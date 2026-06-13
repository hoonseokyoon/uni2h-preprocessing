"""Small RunPod REST client and payload builder.

Tests mock this class and validate payload generation. The client keeps RunPod
credentials outside the database and reads ``RUNPOD_API_KEY`` by default.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://rest.runpod.io/v1"


class RunPodClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerPodSpec:
    name: str
    image_name: str
    server_pod_id: str
    server_port: int
    run_id: str
    worker_token: str
    worker_role: str
    workspace_root: str
    gpu_type_ids: list[str]
    gpu_count: int = 1
    network_volume_id: str | None = None
    data_center_ids: list[str] | None = None
    container_disk_gb: int = 20
    volume_mount_path: str = "/workspace"
    cloud_type: str = "SECURE"
    global_networking: bool = True
    env: dict[str, str] | None = None
    docker_args: str | None = None
    docker_entrypoint: list[str] | None = None
    docker_start_cmd: list[str] | None = None
    ports: list[str] | None = None


class RunPodClient:
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL, timeout_seconds: int = 30):
        self.api_key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def worker_env(
        *,
        server_pod_id: str,
        server_port: int,
        run_id: str,
        worker_token: str,
        worker_role: str = "wsi-preprocess",
        workspace_root: str = "/workspace",
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        env = {
            "SERVER_URL": f"http://{server_pod_id}.runpod.internal:{server_port}",
            "RUN_ID": run_id,
            "WORKER_TOKEN": worker_token,
            "WORKER_ROLE": worker_role,
            "WORKSPACE_ROOT": workspace_root,
        }
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    @staticmethod
    def create_worker_payload(spec: WorkerPodSpec) -> dict[str, Any]:
        env = RunPodClient.worker_env(
            server_pod_id=spec.server_pod_id,
            server_port=spec.server_port,
            run_id=spec.run_id,
            worker_token=spec.worker_token,
            worker_role=spec.worker_role,
            workspace_root=spec.workspace_root,
            extra=spec.env,
        )
        payload: dict[str, Any] = {
            "name": spec.name,
            "imageName": spec.image_name,
            "gpuTypeIds": spec.gpu_type_ids,
            "gpuCount": spec.gpu_count,
            "cloudType": spec.cloud_type,
            "containerDiskInGb": spec.container_disk_gb,
            "volumeMountPath": spec.volume_mount_path,
            "globalNetworking": spec.global_networking,
            "env": env,
        }
        if spec.network_volume_id:
            payload["networkVolumeId"] = spec.network_volume_id
        if spec.data_center_ids:
            payload["dataCenterIds"] = spec.data_center_ids
        if spec.docker_args:
            payload["dockerArgs"] = spec.docker_args
        if spec.docker_entrypoint is not None:
            payload["dockerEntrypoint"] = spec.docker_entrypoint
        if spec.docker_start_cmd is not None:
            payload["dockerStartCmd"] = spec.docker_start_cmd
        if spec.ports:
            payload["ports"] = spec.ports
        return payload

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", payload)

    def list_pods(self) -> dict[str, Any]:
        return self._request("GET", "/pods")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RunPodClientError("RUNPOD_API_KEY is required for RunPod API calls")
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RunPodClientError(f"RunPod API {method} {path} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RunPodClientError(f"RunPod API {method} {path} failed: {exc}") from exc

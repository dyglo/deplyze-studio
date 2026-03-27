import base64
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from roboflow import Roboflow


class RoboflowDatasetService:
    def __init__(self, fallback_api_key: Optional[str] = None):
        self.fallback_api_key = fallback_api_key
        self.cache_root = Path("/tmp/roboflow_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.public_workspaces = [
            "roboflow-universe-projects",
            "roboflow-100",
            "potholedetection-xi4fs",
            "pothole-detection-rdd2022",
        ]

    def resolve_api_key(self, request_api_key: Optional[str]) -> str:
        api_key = request_api_key or self.fallback_api_key
        if not api_key:
            raise RuntimeError("Roboflow API key is required. Provide api_key or set ROBOFLOW_API_KEY in backend .env.")
        return api_key

    def _client(self, api_key: str) -> Roboflow:
        return Roboflow(api_key=api_key)

    def _get_project(self, workspace: str, project: str, api_key: str):
        return self._client(api_key).workspace(workspace).project(project)

    def _workspace_projects(self, api_key: str, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        workspace_obj = self._client(api_key).workspace(workspace) if workspace else self._client(api_key).workspace()
        return list(getattr(workspace_obj, "project_list", []))

    def _latest_version_info(self, project_obj) -> Dict[str, Any]:
        versions = project_obj.get_version_information()
        if not versions:
            raise RuntimeError("No Roboflow versions found for this project.")
        return sorted(versions, key=lambda item: int(Path(item["id"]).name))[-1]

    def _cache_dir(self, workspace: str, project: str, version: int) -> Path:
        return self.cache_root / workspace / project / str(version)

    def _ensure_dataset_cache(self, workspace: str, project: str, version: int, api_key: str) -> Path:
        cache_dir = self._cache_dir(workspace, project, version)
        marker = cache_dir / ".ready"
        if marker.exists():
            return cache_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        project_obj = self._get_project(workspace, project, api_key)
        dataset = project_obj.version(version).download("yolov8", location=str(cache_dir), overwrite=False)
        dataset_location = Path(dataset.location)
        marker.write_text(dataset_location.as_posix(), encoding="utf-8")
        return dataset_location

    def _read_cached_dataset_root(self, workspace: str, project: str, version: int, api_key: str) -> Path:
        cache_dir = self._cache_dir(workspace, project, version)
        marker = cache_dir / ".ready"
        if marker.exists():
            saved_path = Path(marker.read_text(encoding="utf-8").strip())
            if saved_path.exists():
                return saved_path
        return self._ensure_dataset_cache(workspace, project, version, api_key)

    def _load_dataset_yaml(self, dataset_root: Path) -> Dict[str, Any]:
        yaml_path = dataset_root / "data.yaml"
        if not yaml_path.exists():
            return {}
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    def _image_files(self, dataset_root: Path, limit: int = 9) -> List[Path]:
        candidates = []
        for pattern in ("train/images/*", "valid/images/*", "test/images/*", "**/*.jpg", "**/*.jpeg", "**/*.png"):
            candidates.extend(dataset_root.glob(pattern))
        unique = []
        seen = set()
        for path in candidates:
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
            if len(unique) >= limit:
                break
        return unique

    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return encoded, mime

    def _sample_payloads(self, dataset_root: Path, limit: int = 9, embed: bool = True) -> List[Dict[str, Any]]:
        samples = []
        for image_path in self._image_files(dataset_root, limit=limit):
            encoded, mime = self._encode_image(image_path)
            sample = {
                "sample_id": image_path.stem,
                "file_name": image_path.name,
                "mime_type": mime,
                "path": image_path.as_posix(),
            }
            if embed:
                sample["image_base64"] = encoded
                sample["image_url"] = f"data:{mime};base64,{encoded}"
            samples.append(sample)
        return samples

    def search_public_datasets(self, query: str, api_key: str) -> List[Dict[str, Any]]:
        normalized_query = query.strip().lower()
        results = []
        seen = set()
        discovered_workspaces = set()
        try:
            rss = requests.get(
                "https://www.bing.com/search",
                params={"format": "rss", "q": f"site:universe.roboflow.com {query}"},
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            rss.raise_for_status()
            root = ET.fromstring(rss.text)
            for item in root.findall("./channel/item"):
                link = item.findtext("link", default="")
                project_match = re.search(r"https://universe\.roboflow\.com/([^/]+)/([^/?#]+)", link)
                workspace_match = re.search(r"https://universe\.roboflow\.com/([^/?#]+)$", link)
                if project_match:
                    workspace, project = project_match.group(1), project_match.group(2)
                    discovered_workspaces.add(workspace)
                    dataset_id = f"{workspace}/{project}"
                    if dataset_id not in seen:
                        seen.add(dataset_id)
                        results.append({
                            "dataset_id": dataset_id,
                            "name": project.replace("-", " "),
                            "workspace": workspace,
                            "project": project,
                            "version_number": None,
                            "image_count": 0,
                            "class_count": 0,
                            "thumbnail": None,
                            "type": "dataset",
                            "dataset_url": link,
                        })
                elif workspace_match:
                    discovered_workspaces.add(workspace_match.group(1))
        except Exception:
            pass

        for workspace in sorted(discovered_workspaces):
            try:
                for project_meta in self._workspace_projects(api_key, workspace):
                    project = project_meta["id"].split("/")[1]
                    haystack = " ".join([
                        project_meta.get("name", ""),
                        project_meta.get("id", ""),
                        project_meta.get("type", ""),
                        " ".join((project_meta.get("classes") or {}).keys()) if isinstance(project_meta.get("classes"), dict) else " ".join(project_meta.get("classes", [])),
                    ]).lower()
                    if normalized_query not in haystack and not all(token in haystack for token in normalized_query.split() if token):
                        continue
                    dataset_id = f"{workspace}/{project}"
                    if dataset_id in seen:
                        continue
                    seen.add(dataset_id)
                    results.append({
                        "dataset_id": dataset_id,
                        "name": project_meta.get("name", project),
                        "workspace": workspace,
                        "project": project,
                        "version_number": project_meta.get("versions"),
                        "image_count": project_meta.get("images", 0),
                        "class_count": len(project_meta.get("classes", {})),
                        "thumbnail": (project_meta.get("icon") or {}).get("thumb"),
                        "type": project_meta.get("type", "dataset"),
                        "dataset_url": f"https://universe.roboflow.com/{workspace}/{project}",
                    })
            except Exception:
                continue

        for public_workspace in self.public_workspaces:
            try:
                project_list = self._workspace_projects(api_key, public_workspace)
            except Exception:
                continue
            for project_meta in project_list:
                workspace = project_meta["id"].split("/")[0]
                project = project_meta["id"].split("/")[1]
                haystack = " ".join([
                    project_meta.get("name", ""),
                    project_meta.get("id", ""),
                    project_meta.get("type", ""),
                    " ".join((project_meta.get("classes") or {}).keys()) if isinstance(project_meta.get("classes"), dict) else " ".join(project_meta.get("classes", [])),
                ]).lower()
                if normalized_query not in haystack and not all(token in haystack for token in normalized_query.split() if token):
                    continue
                dataset_id = f"{workspace}/{project}"
                if dataset_id in seen:
                    continue
                seen.add(dataset_id)
                results.append({
                    "dataset_id": dataset_id,
                    "name": project_meta.get("name", project),
                    "workspace": workspace,
                    "project": project,
                    "version_number": project_meta.get("versions"),
                    "image_count": project_meta.get("images", 0),
                    "class_count": len(project_meta.get("classes", {})),
                    "thumbnail": (project_meta.get("icon") or {}).get("thumb"),
                    "type": project_meta.get("type", "dataset"),
                    "dataset_url": f"https://universe.roboflow.com/{workspace}/{project}",
                })
        if not results:
            for project_meta in self._workspace_projects(api_key):
                workspace = project_meta["id"].split("/")[0]
                project = project_meta["id"].split("/")[1]
                haystack = " ".join([
                    project_meta.get("name", ""),
                    project_meta.get("id", ""),
                    project_meta.get("type", ""),
                ]).lower()
                if normalized_query not in haystack and not all(token in haystack for token in normalized_query.split() if token):
                    continue
                dataset_id = f"{workspace}/{project}"
                if dataset_id in seen:
                    continue
                seen.add(dataset_id)
                results.append({
                    "dataset_id": dataset_id,
                    "name": project_meta.get("name", project),
                    "workspace": workspace,
                    "project": project,
                    "version_number": project_meta.get("versions"),
                    "image_count": project_meta.get("images", 0),
                    "class_count": len(project_meta.get("classes", {})),
                    "thumbnail": (project_meta.get("icon") or {}).get("thumb"),
                    "type": project_meta.get("type", "dataset"),
                    "dataset_url": f"https://universe.roboflow.com/{workspace}/{project}",
                })
        if not results and "/" in query:
            workspace, project = query.split("/", 1)
            project_obj = self._get_project(workspace, project, api_key)
            latest = self._latest_version_info(project_obj)
            results.append({
                "dataset_id": f"{workspace}/{project}",
                "name": project_obj.name,
                "workspace": workspace,
                "project": project,
                "version_number": int(Path(latest["id"]).name),
                "image_count": latest.get("images", project_obj.images),
                "class_count": len(project_obj.classes),
                "thumbnail": None,
                "type": project_obj.type,
                "dataset_url": f"https://universe.roboflow.com/{workspace}/{project}",
            })
        return results[:8]

    def dataset_detail(self, workspace: str, project: str, api_key: str) -> Dict[str, Any]:
        project_obj = self._get_project(workspace, project, api_key)
        latest = self._latest_version_info(project_obj)
        version_number = int(Path(latest["id"]).name)
        dataset_root = self._read_cached_dataset_root(workspace, project, version_number, api_key)
        dataset_yaml = self._load_dataset_yaml(dataset_root)
        classes = dataset_yaml.get("names") or project_obj.classes or []
        if isinstance(classes, dict):
            classes = list(classes.values())
        samples = self._sample_payloads(dataset_root, limit=9, embed=True)

        return {
            "dataset_id": f"{workspace}/{project}",
            "name": project_obj.name,
            "workspace": workspace,
            "project": project,
            "type": project_obj.type,
            "version": version_number,
            "image_count": latest.get("images", project_obj.images),
            "class_count": len(classes),
            "classes": classes,
            "samples": samples,
            "dataset_url": f"https://universe.roboflow.com/{workspace}/{project}",
        }

    def inferable_samples(self, workspace: str, project: str, version: Optional[int], api_key: str, limit: int = 4) -> List[Dict[str, Any]]:
        if version is None:
            project_obj = self._get_project(workspace, project, api_key)
            version = int(Path(self._latest_version_info(project_obj)["id"]).name)
        dataset_root = self._read_cached_dataset_root(workspace, project, int(version), api_key)
        return self._sample_payloads(dataset_root, limit=limit, embed=True)

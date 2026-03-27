import base64
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import server


client = TestClient(server.app)


def test_roboflow_search_endpoint(monkeypatch):
    monkeypatch.setattr(server.roboflow_service, "resolve_api_key", lambda key: "resolved-key")
    monkeypatch.setattr(
        server.roboflow_service,
        "search_public_datasets",
        lambda query, api_key: [
            {
                "dataset_id": "ws/project",
                "name": "Sample Dataset",
                "workspace": "ws",
                "project": "project",
                "version_number": 3,
                "image_count": 42,
                "class_count": 5,
                "thumbnail": None,
                "type": "object-detection",
                "dataset_url": "https://universe.roboflow.com/ws/project",
            }
        ],
    )

    response = client.get("/api/roboflow/search?q=sample")
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["dataset_id"] == "ws/project"


def test_roboflow_dataset_detail_endpoint(monkeypatch):
    monkeypatch.setattr(server.roboflow_service, "resolve_api_key", lambda key: "resolved-key")
    monkeypatch.setattr(
        server.roboflow_service,
        "dataset_detail",
        lambda workspace, project, api_key: {
            "dataset_id": f"{workspace}/{project}",
            "name": "Sample Dataset",
            "workspace": workspace,
            "project": project,
            "type": "object-detection",
            "version": 2,
            "image_count": 12,
            "class_count": 3,
            "classes": ["car", "bus", "truck"],
            "samples": [{"sample_id": "one", "image_url": "data:image/jpeg;base64,AA=="}],
        },
    )

    response = client.get("/api/roboflow/dataset/ws/project")
    assert response.status_code == 200
    assert response.json()["class_count"] == 3


def test_roboflow_infer_detect_endpoint(monkeypatch):
    monkeypatch.setattr(server.roboflow_service, "resolve_api_key", lambda key: "resolved-key")
    monkeypatch.setattr(
        server.roboflow_service,
        "inferable_samples",
        lambda workspace, project, version, api_key: [
            {
                "sample_id": "sample-1",
                "file_name": "sample.jpg",
                "mime_type": "image/jpeg",
                "image_base64": base64.b64encode(b"image").decode("utf-8"),
                "image_url": "data:image/jpeg;base64,aW1hZ2U=",
            }
        ],
    )
    monkeypatch.setattr(server, "base64_to_numpy", lambda value: "image-array")
    monkeypatch.setattr(
        server.ml_service,
        "detect_logos_in_image",
        lambda image: {
            "inference_time": 0.123,
            "detections": [{"class_name": "car", "confidence": 0.9, "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "width": 2, "height": 2}}],
        },
    )

    response = client.post(
        "/api/roboflow/infer",
        json={
            "dataset_id": "ws/project",
            "workspace": "ws",
            "project": "project",
            "version": 1,
            "task": "detect",
            "model": "yolo11n",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["detections"][0]["class_name"] == "car"


def test_track_endpoint(monkeypatch, tmp_path):
    def fake_track(video_path, output_path, model_name=None, tracker="botsort"):
        with open(output_path, "wb") as handle:
            handle.write(b"video")
        return {
            "success": True,
            "processed_frames": 2,
            "total_detections": 3,
            "processing_time": 0.5,
            "fps": 24,
            "resolution": "320x240",
            "avg_detections_per_frame": 1.5,
            "tracker": tracker,
            "model_used": model_name or "yolo11n",
            "total_tracks": 2,
            "frame_results": [{"frame_index": 1, "detections": [{"track_id": 7}]}],
        }

    monkeypatch.setattr(server.ml_service, "track_video_file", fake_track)

    response = client.post(
        "/api/track",
        json={
            "video": base64.b64encode(b"fake-video").decode("utf-8"),
            "model": "yolo11n",
            "tracker": "bytetrack",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tracker"] == "bytetrack"
    assert body["frame_results"][0]["detections"][0]["track_id"] == 7


def test_model_convert_endpoint(monkeypatch):
    monkeypatch.setattr(
        server.ml_service,
        "convert_pt_to_onnx",
        lambda path, task="detect": {
            "success": True,
            "onnx_model": base64.b64encode(b"onnx-bytes").decode("utf-8"),
            "filename": "model.onnx",
        },
    )

    response = client.post(
        "/api/model/convert?task=segment",
        files={"file": ("custom.pt", BytesIO(b"weights"), "application/octet-stream")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "model.onnx"

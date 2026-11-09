import json
from datetime import datetime

import pytest

import tools


def test_catalog_filters_category_and_inclusive_price(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    products = [
        {"category": "xe_dien", "price_vnd": 600},
        {"category": "xe_dien", "price_vnd": 601},
        {"category": "du_lich", "price_vnd": 100},
    ]
    (tmp_path / "product_catalog.json").write_text(json.dumps(products), encoding="utf-8")
    assert tools.search_product_catalog("xe_dien", 600) == products[:1]
    assert tools.search_product_catalog("du_lich") == products[2:]
    assert tools.search_product_catalog("xe_dien", 0) == []


def test_catalog_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    assert tools.search_product_catalog("xe_dien") == []


def test_tickets_are_persisted_without_losing_existing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    first = tools.submit_support_ticket("Nguyễn An", "Lỗi pin")
    assert first["status"] == "open"
    assert first["priority"] == "medium"
    assert first["ticket_id"].startswith("TK-" + datetime.now().strftime("%Y%m%d") + "-")
    datetime.fromisoformat(first["created_at"])
    second = tools.submit_support_ticket("Bình", "Lỗi sạc", "high")
    assert second["ticket_id"] != first["ticket_id"]
    assert second["priority"] == "high"
    saved = json.loads((tmp_path / "support_tickets.json").read_text(encoding="utf-8"))
    assert saved == [first, second]
    assert saved[0]["customer_name"] == "Nguyễn An"
    assert saved[0]["issue_description"] == "Lỗi pin"


def test_corrupt_tickets_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    path = tmp_path / "support_tickets.json"
    path.write_text("broken JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        tools.submit_support_ticket("An", "Lỗi pin")
    assert path.read_text(encoding="utf-8") == "broken JSON"


def test_schemas_describe_callable_tools():
    definitions = {item["name"]: item for item in tools.TOOL_DEFINITIONS}
    assert definitions.keys() == tools.TOOL_MAP.keys()
    for name, definition in definitions.items():
        assert definition["description"]
        parameters = definition["parameters"]
        assert parameters["type"] == "object"
        assert set(parameters["required"]) <= parameters["properties"].keys()
    assert definitions["search_product_catalog"]["parameters"]["required"] == ["category"]
    assert definitions["submit_support_ticket"]["parameters"]["required"] == [
        "customer_name", "issue_description"
    ]

import json

import pytest

import tools
from template import ChatbotBaseline, ToolCallingAgent


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RAW_DATA_DIR", str(tmp_path))
    products = [
        {"name": "VF 3", "category": "xe_dien", "price_vnd": 315000000},
        {"name": "VF 8", "category": "xe_dien", "price_vnd": 1129000000},
        {"name": "Vinpearl", "category": "du_lich", "price_vnd": 3500000},
    ]
    (tmp_path / "product_catalog.json").write_text(json.dumps(products), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("budget", ["600 triệu", "0,6 tỷ", "600.000.000 đồng"])
def test_catalog_budget_and_trace(data_dir, budget):
    result = ToolCallingAgent().run(f"Xem xe điện giá dưới {budget}")
    assert result["status"] == "completed"
    assert "VF 3" in result["answer"]
    assert "VF 8" not in result["answer"]
    assert result["iterations"] == 1
    assert any(step["step"] == "Observation" for step in result["trace"])


def test_multiple_tools_and_ticket_details(data_dir):
    result = ToolCallingAgent().run(
        "Xem du lịch dưới 6 triệu. Tôi tên Nguyễn An, xe VF 8 bị lỗi pin, cần xử lý gấp."
    )
    assert result["status"] == "completed"
    assert result["iterations"] == 2
    assert "Vinpearl" in result["answer"]
    assert "TK-" in result["answer"]
    ticket = json.loads((data_dir / "support_tickets.json").read_text(encoding="utf-8"))[0]
    assert ticket["customer_name"] == "Nguyễn An"
    assert ticket["priority"] == "high"
    assert "lỗi pin" in ticket["issue_description"]


def test_missing_name_asks_without_writing_ticket(data_dir):
    result = ToolCallingAgent().run("Xe VinFast bị lỗi pin, cần hỗ trợ")
    assert result["status"] == "needs_information"
    assert not (data_dir / "support_tickets.json").exists()


def test_faq_does_not_fabricate_warranty_or_create_ticket(data_dir):
    result = ToolCallingAgent().run("Chính sách bảo hành pin VinFast bao lâu?")
    assert result["status"] == "completed"
    assert "bảo hành" in result["answer"].lower()
    assert "10 năm" not in result["answer"]
    assert not any(step["step"] == "Action" for step in result["trace"])


def test_iteration_limit_retains_partial_results(data_dir):
    result = ToolCallingAgent(max_iterations=1).run(
        "Xem xe điện. Tôi tên An, xe bị lỗi pin, cần hỗ trợ."
    )
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 1
    assert "VF 3" in result["answer"]
    assert not (data_dir / "support_tickets.json").exists()


def test_tool_error_is_reported(data_dir):
    (data_dir / "product_catalog.json").write_text("invalid", encoding="utf-8")
    result = ToolCallingAgent().run("Xem xe điện")
    assert result["status"] == "error"
    assert "chưa" in result["answer"].lower()


def test_out_of_scope_and_baseline_do_not_call_tools(data_dir):
    result = ToolCallingAgent().run("Viết bài văn về lịch sử thế giới")
    assert "Vingroup" in result["answer"]
    assert not any(step["step"] == "Action" for step in result["trace"])
    assert ChatbotBaseline().query("Xem xe VinFast")["tool_calls"] == []


def test_empty_results_and_trace_reset(data_dir):
    agent = ToolCallingAgent()
    first = agent.run("Xem xe điện dưới 200 triệu")
    assert "không tìm thấy" in first["answer"].lower()
    agent.run("Xin chào")
    assert agent.trace is not first["trace"]


def test_zero_iterations_does_not_call_tools(data_dir):
    result = ToolCallingAgent(max_iterations=0).run("Xem xe điện")
    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 0


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("trên 600 triệu", ["Above"]),
        ("dưới 600 triệu", ["Below"]),
        ("tối đa 600 triệu", ["Below", "Equal"]),
        ("ít nhất 600 triệu", ["Equal", "Above"]),
        ("trên 0,6 tỷ", ["Above"]),
        ("trên 600.000.000 đồng", ["Above"]),
        ("từ 600 triệu đến 700 triệu", ["Equal", "Above"]),
        ("trên 600 triệu và dưới 700 triệu", []),
    ],
)
def test_price_comparison_boundaries(data_dir, condition, expected):
    products = [
        {"name": "Below", "category": "xe_dien", "price_vnd": 500000000},
        {"name": "Equal", "category": "xe_dien", "price_vnd": 600000000},
        {"name": "Above", "category": "xe_dien", "price_vnd": 700000000},
    ]
    (data_dir / "product_catalog.json").write_text(json.dumps(products), encoding="utf-8")
    result = ToolCallingAgent().run(f"Tôi muốn xem xe điện VinFast giá {condition}")
    assert result["status"] == "completed"
    for product in products:
        assert (product["name"] in result["answer"]) == (product["name"] in expected)
    if not expected:
        assert "không tìm thấy" in result["answer"].lower()

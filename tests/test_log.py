from app.log import mask_phone, mask_phone_numbers


def test_masks_common_indian_formats():
    for raw in ("+919581283456", "+91 95812 83456", "9581283456", "+91-787-422-3333"):
        masked = mask_phone(f"call {raw} now")
        assert masked.startswith("call XXXXXXXX") and masked.endswith(" now"), masked
        assert raw not in masked


def test_leaves_prices_and_pincodes_alone():
    text = "Price 1049000 in 500072"
    assert mask_phone(text) == text


def test_processor_masks_nested_fields():
    event = {"event": "lead saved", "lead": {"phone": "9876543210", "city": "Hyderabad"}}
    out = mask_phone_numbers(None, "info", event)
    assert out["lead"]["phone"] == "XXXXXXXX10"
    assert out["lead"]["city"] == "Hyderabad"

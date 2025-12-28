from __future__ import annotations
from typing import Final


FEATURES_DROP_ALWAYS: Final[list[str]] = [
    "availability",
]

SALES_TEST_COLS: Final[list[str]] = [
    "unique_id",
    "date",
    "warehouse",
    "total_orders",
    "sell_price_main",
    "type_0_discount",
    "type_1_discount",
    "type_2_discount",
    "type_3_discount",
    "type_4_discount",
    "type_5_discount",
    "type_6_discount",
]

SALES_TRAIN_COLS: Final[list[str]] = SALES_TEST_COLS + ["sales", "availability"]

DISCOUNT_COLS: Final[list[str]] = [
    c for c in SALES_TEST_COLS if c.startswith("type_") and c.endswith("_discount")
]

SALES_DTYPES: Final[dict[str, str]] = {
    "unique_id": "int32",
    "warehouse": "category",
    "total_orders": "float32",
    "sales": "float32",
    "availability": "float32",
    "sell_price_main": "float32",
    "type_0_discount": "float32",
    "type_1_discount": "float32",
    "type_2_discount": "float32",
    "type_3_discount": "float32",
    "type_4_discount": "float32",
    "type_5_discount": "float32",
    "type_6_discount": "float32",
}

CALENDAR_COLS: Final[list[str]] = [
    "date",
    "holiday_name",
    "holiday",
    "shops_closed",
    "winter_school_holidays",
    "school_holidays",
    "warehouse",
]

CALENDAR_DTYPES: Final[dict[str, str]] = {
    "holiday_name": "category",
    "holiday": "int8",
    "shops_closed": "int8",
    "winter_school_holidays": "int8",
    "school_holidays": "int8",
    "warehouse": "category",
}

INVENTORY_COLS: Final[list[str]] = [
    "unique_id",
    "product_unique_id",
    "name",
    "L1_category_name_en",
    "L2_category_name_en",
    "L3_category_name_en",
    "L4_category_name_en",
    "warehouse",
]

INVENTORY_DTYPES: Final[dict[str, str]] = {
    "unique_id": "int32",
    "product_unique_id": "int32",
    "name": "category",
    "L1_category_name_en": "category",
    "L2_category_name_en": "category",
    "L3_category_name_en": "category",
    "L4_category_name_en": "category",
    "warehouse": "category",
}




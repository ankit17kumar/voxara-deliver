"""Order store and OMS adapter.

`MockOMS` keeps orders in SQLite so the MVP runs anywhere. A real
integration (Shiprocket, Unicommerce, ClickPost, a brand's own OMS)
implements the same four methods: get_order, list_orders, apply_outcome,
and events. Every write is recorded as an event so the dashboard can show
"verified update in OMS", the deck's action-completion KPI.
"""
from __future__ import annotations

import json
import random
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
METRO_CITIES = {"Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Pune"}


@dataclass
class Order:
    order_id: str
    brand: str
    customer_name: str
    phone: str
    lang_pref: str
    city: str
    pincode: str
    address_line: str
    landmark: str
    product: str
    amount: float
    payment: str            # COD | PREPAID
    status: str             # CONFIRMATION_PENDING | CONFIRMED | NDR | CANCELLED | RTO | REATTEMPT_SCHEDULED
    ndr_reason: str = ""
    attempts: int = 0
    reattempt_date: str = ""
    reattempt_slot: str = ""
    notes: str = ""

    @property
    def is_metro(self) -> bool:
        return self.city in METRO_CITIES

    @property
    def masked_phone(self) -> str:
        return "XXXXXX" + self.phone[-4:]

    def to_dict(self):
        d = asdict(self)
        d["phone"] = self.masked_phone
        d["is_metro"] = self.is_metro
        return d


class MockOMS:
    def __init__(self, db_path: str | Path = ":memory:"):
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, data TEXT);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, order_id TEXT, action TEXT, payload TEXT);
        """)

    def upsert(self, order: Order):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO orders VALUES (?, ?)",
                            (order.order_id, json.dumps(asdict(order))))
            self.db.commit()

    def get_order(self, order_id: str) -> Order | None:
        row = self.db.execute("SELECT data FROM orders WHERE order_id=?", (order_id,)).fetchone()
        return Order(**json.loads(row["data"])) if row else None

    def list_orders(self) -> list[Order]:
        return [Order(**json.loads(r["data"])) for r in
                self.db.execute("SELECT data FROM orders ORDER BY order_id")]

    def apply_outcome(self, order_id: str, action: str, **fields) -> Order:
        """Write a call outcome back to the OMS. Returns the updated order."""
        order = self.get_order(order_id)
        if order is None:
            raise KeyError(order_id)
        for k, v in fields.items():
            if hasattr(order, k):
                setattr(order, k, v)
        self.upsert(order)
        with self._lock:
            self.db.execute("INSERT INTO events (ts, order_id, action, payload) VALUES (?,?,?,?)",
                            (datetime.now(IST).isoformat(timespec="seconds"), order_id, action,
                             json.dumps(fields, default=str)))
            self.db.commit()
        return order

    def events(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]


def seed(oms: MockOMS, n: int = 24, rnd: random.Random | None = None):
    """Synthetic Indian e-commerce orders. Clearly fake: no real people."""
    rnd = rnd or random.Random(7)
    names = ["Priya Sharma", "Rahul Verma", "Ananya Iyer", "Mohammed Irfan", "Sneha Patil",
             "Arjun Reddy", "Kavya Nair", "Vikram Singh", "Pooja Gupta", "Ravi Kumar",
             "Meera Joshi", "Suresh Yadav"]
    places = [("Bengaluru", "560034", "Koramangala 4th Block"), ("Mumbai", "400053", "Andheri West"),
              ("Delhi", "110024", "Lajpat Nagar"), ("Jaipur", "302017", "Malviya Nagar"),
              ("Lucknow", "226010", "Gomti Nagar"), ("Indore", "452010", "Vijay Nagar"),
              ("Patna", "800001", "Boring Road"), ("Pune", "411038", "Kothrud")]
    products = [("Cotton kurta set", 1299), ("Wireless earbuds", 1899), ("Ayurvedic hair oil", 449),
                ("Kids school bag", 799), ("Non-stick kadai", 1099), ("Running shoes", 2499),
                ("Face serum", 599), ("Bedsheet double", 899)]
    ndr_reasons = ["Customer not available", "Customer refused", "Address incomplete",
                   "Phone unreachable", "Customer asked to reschedule", "COD amount not ready"]
    brands = ["UrbanKurta", "SoundBee", "VedaGlow", "HomeChef"]
    for i in range(n):
        city, pin, area = rnd.choice(places)
        prod, amt = rnd.choice(products)
        is_ndr = i % 2 == 0
        name = rnd.choice(names)
        oms.upsert(Order(
            order_id=f"VX{10231 + i}", brand=rnd.choice(brands), customer_name=name,
            phone=f"9{rnd.randint(100000000, 999999999)}",
            lang_pref=rnd.choice(["hi-en", "hi-en", "en"]), city=city, pincode=pin,
            address_line=f"{rnd.randint(1, 400)}, {area}" if rnd.random() > 0.2 else area,
            landmark="" if rnd.random() > 0.5 else "Near SBI ATM",
            product=prod, amount=float(amt), payment="COD",
            status="NDR" if is_ndr else "CONFIRMATION_PENDING",
            ndr_reason=rnd.choice(ndr_reasons) if is_ndr else "",
            attempts=rnd.randint(1, 2) if is_ndr else 0))

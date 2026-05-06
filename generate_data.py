import polars as pl
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
random.seed(42)
clients =[]
for i in range (20):
    clients.append({"name": fake.company(), "late_probability": random.uniform(0.1, 0.9)})

def generate_invoices(invoice_id, client):
    issue_date = fake.date_between(start_date="-1y", end_date="today")
    due_date = issue_date + timedelta(days=random.choice([15, 30, 45, 60]))
    will_pay_late = random.random() < client["late_probability"]
    if will_pay_late:
            paid_date = due_date + timedelta(days=random.randint(1, 45))
    else:
        paid_date = due_date - timedelta(days=random.randint(0, 10))
    return {
        "invoice_id": invoice_id,
        "client_name": client["name"],
        "amount": round(random.uniform(500, 50000), 2),
        "issue_date": issue_date,
        "due_date": due_date,
        "paid_date": paid_date,
    }

invoices = []
for i in range(1000):
    invoices.append(generate_invoices(i, random.choice(clients)))

df = pl.DataFrame(invoices)
print(df)

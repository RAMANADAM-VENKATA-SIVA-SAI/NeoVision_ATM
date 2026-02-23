import json
import random

first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "William", "Elizabeth", 
               "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", 
               "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra",
               "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
              "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]

data = {}

for i in range(1, 151):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    acc_num = f"4000-5000-{1000 + i}"
    balance = random.randint(1000, 100000)
    
    data[acc_num] = {
        "account_holder": name,
        "account_number": acc_num,
        "balance": balance,
        "transactions": [
            {"date": "2026-02-01", "type": "Deposit", "amount": balance}
        ]
    }

with open('c:/Users/HP/.gemini/antigravity/scratch/neovision-atm/evo/data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Generated 150 members in data.json")

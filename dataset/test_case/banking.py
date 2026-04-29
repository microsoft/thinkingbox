# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common import Judge, TestContext

"""!
scenario: retail_banking
"""


def test_transfer_and_balance(x: TestContext, judge: Judge):
    """!
    query: |
        I want to transfer $100 from my checking account.
    user_context: |
        You are interested in transferring money from your checking account to your savings account. You don't know any checking or savings account number.
    """
    # transfer should be 100 from checking to savings
    assert x.effects["online_banking"].get("transactions", [])
    assert (
        x.effects["online_banking"]["transactions"][0].items()
        >= {
            "transaction_type": "transfer",
            "from_account": "599521725",
            "to_account": "843847393",
            "amount": 100.0,
        }.items()
    )
    # savings account balance should be 5100.00
    assert "843847393" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["843847393"] == {
        "account_type": "savings",
        "account_number": "843847393",
        "balance": 5100.0,
    }
    # checking account balance should be 900.00
    assert "599521725" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["599521725"] == {
        "account_type": "checking",
        "account_number": "599521725",
        "balance": 900.0,
    }
    assert judge.text_yesno(x.response, "Does the message confirm the transfer?")


def test_get_balance_savings(x: TestContext, judge: Judge):
    """!
    query:
        What's the balance of my savings account?
    user_context: |
        You are a user of an online banking system and want to know the current balance of your savings account.
    """
    # the assistant should explicitly mention (≈ confirm) the $5000 balance
    assert judge.text_yesno(
        x.response, "Does the message confirm the balance is 5000.00?"
    )


def test_transfer_savings_to_investment(x: TestContext, judge: Judge):
    """!
    query: |
        Please move $2500 from my savings to my investment account.
    user_context: |
        You're an online-banking customer who would like to invest additional funds by moving money from savings into your investment account.
    """
    # transfer of 2500 from savings → investment
    assert x.effects["online_banking"].get("transactions", [])
    assert (
        x.effects["online_banking"]["transactions"][0].items()
        >= {
            "transaction_type": "transfer",
            "from_account": "843847393",
            "to_account": "691240002",
            "amount": 2500.0,
        }.items()
    )
    # new balances
    assert "843847393" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["843847393"] == {
        "account_type": "savings",
        "account_number": "843847393",
        "balance": 2500.0,
    }
    assert "691240002" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["691240002"] == {
        "account_type": "investment",
        "account_number": "691240002",
        "balance": 17500.0,
    }
    # assistant should confirm the transfer
    assert judge.text_yesno(
        x.response,
        "Does the message confirm a transfer from savings to investment account?",
    )


def test_pay_bill_and_transfer(x: TestContext, judge: Judge):
    """!
    query: |
        Pay my $120 electric bill, then transfer $300 from checking to savings.
    user_context: |
        As an online-banking user you want to (1) pay your electric bill from your checking account, and (2) move some money into savings for later.
    """
    # --- bill payment section -------------------------------------------------
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "599521725",
            "payee_account": "PAY1001",
            "amount": 120.0,
        }.items()
    )
    # --- transfer section -----------------------------------------------------
    assert x.effects["online_banking"].get("transactions", [])
    assert (
        x.effects["online_banking"]["transactions"][0].items()
        >= {
            "transaction_type": "transfer",
            "from_account": "599521725",
            "to_account": "843847393",
            "amount": 300.0,
        }.items()
    )
    # --- expected balances ----------------------------------------------------
    assert "599521725" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["599521725"] == {
        "account_type": "checking",
        "account_number": "599521725",
        "balance": 580.0,
    }
    assert "843847393" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["843847393"] == {
        "account_type": "savings",
        "account_number": "843847393",
        "balance": 5300.0,
    }
    # assistant must confirm *both* actions
    assert judge.text_yesno(
        x.response,
        "Does the message confirm two actions, "
        "one related to electric company or bill, one related to a transfer?",
    )


def test_pay_bill_1(x: TestContext, judge: Judge):
    """!
    query: |
        Pay $200 to my Watercompany from my account ?
    user_context: |
        Any payment should come from checking account
    """
    # --- Water company bill payment section
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "599521725",
            "payee_account": "PAY1003",
            "amount": 200.0,
        }.items()
    )
    # --- expected balances
    assert "599521725" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["599521725"] == {
        "account_type": "checking",
        "account_number": "599521725",
        "balance": 800.0,
    }
    # assistant confirm the payment was made
    assert judge.text_yesno(
        x.response,
        "Does the message confirm the payment of $200 to the Water Company has been successfully scheduled from your checking account ?",
    )


def test_check_balance_1(x: TestContext, judge: Judge):
    """!
    query: |
        What is the balance of my account?
    user_context: |
        You'd like to know the balance of your investment account.
    """
    # Check the balance of investment
    assert judge.text_yesno(
        x.response, "Does the message confirm the balance is 15000 ?"
    )


def test_pay_mortgage(x: TestContext, judge: Judge):
    """!
    query: |
        Can you pay my Mortgage ?
    user_context: |
        Use Investment account to pay the Mortgage for an amount of $1000
    """
    # Check the balance of investment
    assert "691240002" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["691240002"] == {
        "account_type": "investment",
        "account_number": "691240002",
        "balance": 14000.0,
    }
    assert judge.text_yesno(
        x.response,
        "Does the message confirm the payment of $1000 from your Investment account to your mortgage lender has been successfully scheduled?",
    )


def test_pay_bill_2(x: TestContext, judge: Judge):
    """!
    query: |
        pay $200 for Internet use.
    user_context: |
        Use checking account for paying Internet bill
    """
    # --- expected balances
    assert "599521725" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["599521725"] == {
        "account_type": "checking",
        "account_number": "599521725",
        "balance": 800.0,
    }
    # --- Internet bill payment section
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "599521725",
            "payee_account": "PAY1002",
            "amount": 200.0,
        }.items()
    )
    # Verify the final response if correct
    assert judge.text_yesno(
        x.response, "Does the message confirm payment of $200 to Internet provider ?"
    )


def test_pay_bill_3(x: TestContext, judge: Judge):
    """!
    query: |
        Pay my electric bill from my account
    user_context: |
        Pay $100 from Checking account
    """
    # --- bill payment section -------------------------------------------------
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "599521725",
            "payee_account": "PAY1001",
            "amount": 100.0,
        }.items()
    )
    # --- expected balances ----------------------------------------------------
    assert "599521725" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["599521725"] == {
        "account_type": "checking",
        "account_number": "599521725",
        "balance": 900.0,
    }

    assert judge.text_yesno(
        x.response, "Does the message confirm payment of $100 towards ElectricCompany ?"
    )


def test_check_balance_2(x: TestContext, judge: Judge):
    """!
    query: |
        What's my balance?
    user_context: |
        Use Investment account
    """
    # Check the balance of investment
    assert judge.text_yesno(
        x.response,
        "Does the message confirm the balance is 15000 ?",
    )


def test_pay_mortgage_1(x: TestContext, judge: Judge):
    """!
    query: |
        Pay $200 in interest on a mortgage?
    user_context: |
        Use Saving account and to pay the mortgage
    """
    # TODO this may be ambiguous because by convention any payments are made from
    # the checking account, it is unusual to pay directly from a saving account
    # (and often not possible)

    # Pay to mortgage account
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "843847393",
            "payee_account": "PAY1004",
            "amount": 200.0,
        }.items()
    )
    # --- expected balances ----------------------------------------------------
    assert "843847393" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["843847393"] == {
        "account_type": "savings",
        "account_number": "843847393",
        "balance": 4800.0,
    }
    # Check the final reponse.
    assert judge.text_yesno(
        x.response, "Does the message confirm $200 payment to MortgageLender ?"
    )


def test_pay_mortgage_bot_instructions(x: TestContext, judge: Judge):
    """!
    query: |
        Can you please pay $200 to my mortgage?
    user_context: |
        If asked, confirm to proceed with the payment.
    bot_instructions: |
        User's recent payment history:
        (1 month ago) Paid $200 from savings (843847393) to MortgageLender (PAY1004)
        (2 months ago) Paid $200 from savings (843847393) to MortgageLender (PAY1004)
        (3 months ago) Paid $200 from savings (843847393) to MortgageLender (PAY1004)
    tags: [skip]
    """
    # This test only demonstrates the usage of bot_instructions
    # Pay to mortgage account
    assert x.effects["online_banking"].get("scheduled_bills", [])
    assert (
        x.effects["online_banking"]["scheduled_bills"][0].items()
        >= {
            "user_account": "843847393",
            "payee_account": "PAY1004",
            "amount": 200.0,
        }.items()
    )
    # --- expected balances ----------------------------------------------------
    assert "843847393" in x.effects["online_banking"]["accounts"]
    assert x.effects["online_banking"]["accounts"]["843847393"] == {
        "account_type": "savings",
        "account_number": "843847393",
        "balance": 4800.0,
    }
    # Check the final reponse.
    assert judge.text_yesno(
        x.response, "Does the message confirm $200 payment to MortgageLender ?"
    )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common import Judge, TestContext

"""!
scenario: retail_banking_email
"""


def test_transfer_and_sendemail(x: TestContext, judge: Judge):
    """!
    query: |
         Transfer $100 from my checking account and then send an email with confirmation?
    user_context: |
        I'd like to transfer to savings account, and the confirmation email should be sent to john@example.com
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

    effects_list = x.effects["email_system"]["effects"]
    send_ops = [e for e in effects_list if e.get("op") == "send_message"]
    assert len(send_ops) == 1
    # Check if the email was sent to John
    assert send_ops[0].get("to") == ["john@example.com"]
    assert judge.text_yesno(
        send_ops[0].get("body"),
        "Does the message confirm the transfer is done?",
    )

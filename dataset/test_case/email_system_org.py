# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common import Judge, TestContext

"""!
scenario: email_system_org
"""


def test_email_manager_of_manager(x: TestContext, judge: Judge):
    """!
    query: |
        Email the manager of my manager the following "Are you free for a meeting regarding Project A on Thursday?"
    """
    effects_list = x.effects["email_system"]["effects"]

    send_ops = [e for e in effects_list if e.get("op") == "send_message"]
    assert len(send_ops) == 1

    # Verify the email was sent to Carol VP (manager of manager)
    assert send_ops[0].get("to") == ["carol@company.com"]
    assert send_ops[0].get("cc") == []
    assert judge.text_yesno(
        send_ops[0].get("body"),
        "Does the message ask whether the recipient is free for a meeting regarding Project A on Thursday? ",
    )

    assert send_ops[0].get("_from") == "alice@company.com"


def test_organizational_structure_contact_lookup(x: TestContext, judge: Judge):
    """!
    query: |
        Who is my manager's manager?
    """
    # This test verifies that the contact lookup works correctly
    # Alice's manager is Bob (contact-001), Bob's manager is Carol (contact-002)
    effects_list = x.effects["email_system"]["effects"]

    # Should have get_contact operations to traverse the hierarchy
    get_contact_ops = [e for e in effects_list if e.get("op") == "get_contact"]
    assert len(get_contact_ops) >= 1
    assert judge.text_yesno(
        x.response, "Does the message say Carol is the manager's manager?"
    )


def test_contact_hierarchy_navigation(x: TestContext, judge: Judge):
    """!
    query: |
        List all my coworkers who report to the same manager as me
    """
    effects_list = x.effects["email_system"]["effects"]

    # Should list contacts and potentially get specific contact details
    list_contacts_ops = [e for e in effects_list if e.get("op") == "list_contacts"]
    assert len(list_contacts_ops) >= 1
    assert judge.text_yesno(
        x.response, "Does the message list David, Eve and Frank as colleagues?"
    )

"""``iris email demo``: the email assistant end to end on a synthetic mailbox.

No credentials, no network, no model server, and never the owner's profile. The pieces:

* ``corpus.py`` -- the seeded generator of the 200-email demo mailbox (``corpus.json``);
* ``provider.py`` -- the ``demo`` ``MailProvider`` that serves it;
* ``fake_model.yaml`` -- the script the scripted fake model answers from;
* ``home.py`` -- the isolated demo home and the environment a run gets;
* ``run.py`` -- one run: fetch, judge, first digest, and what IRIS did.
"""

"""Connecting the owner's external accounts from the web console.

``google`` reconnects the Google accounts the gmail, calendar and file_organizer plugins
read (their OAuth flow on the server, with a Web client), so a revoked token is fixed
from the phone instead of by a Mac login and a copy to the VM. Each plugin registers
its own provider here; the shared routes mount once.
"""

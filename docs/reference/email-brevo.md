# Email verification (Brevo)

How verification codes are sent when advocates sign in, and how to configure Brevo for production.

---

## Default: terminal only (no SMTP)

**If `ILGA_SMTP_HOST` is not set** (or all `ILGA_SMTP_*` are commented out in `.env`), the app **does not send email**. Instead, it logs the 6-digit verification code to the **terminal** in a box. That way you can develop and test the sign-in flow without configuring SMTP or hitting Brevo.

- **Local dev:** Leave `ILGA_SMTP_*` commented out in `.env` → codes appear in the terminal only.
- **When you want to test real email:** Uncomment and set the Brevo vars in `.env`, then restart.
- **Production:** Set `ILGA_SMTP_*` in your host’s environment so codes are sent via Brevo.

---

## Where to put your SMTP key

**Never commit your real API key to the repo.**

- **Local:** Put Brevo vars in `.env` (gitignored). Comment them out for terminal-only dev.
- **Production:** Set the same vars in your host’s environment or secrets.

---

## Brevo setup

1. Sign up at [Brevo](https://www.brevo.com/).
2. **Settings → SMTP & API → SMTP** tab. Copy **both** the **SMTP login** (username) and **SMTP key** (password).
3. `ILGA_SMTP_USER` must be the **SMTP login** from that page — not your Brevo account email.
4. Set in `.env` or production env:

| Variable | Value |
|----------|--------|
| `ILGA_SMTP_HOST` | `smtp-relay.brevo.com` |
| `ILGA_SMTP_PORT` | `587` |
| `ILGA_SMTP_USER` | SMTP login from Brevo (e.g. `…@smtp-brevo.com`) |
| `ILGA_SMTP_PASS` | SMTP key from Brevo |
| `ILGA_SMTP_FROM` | e.g. `noreply@yourdomain.com` (after domain auth) |
| `ILGA_SMTP_TLS` | `1` |

See [Environment variables](environment-variables.md) and [Deployment](deployment.md).

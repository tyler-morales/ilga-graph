/**
 * Hero auth strip and inline sign-in: shared by home, advocacy, and poll fragment.
 * Provides getCsrfToken, window._ilgaUserEmail, window._pendingAuthEmail, updateAuthStrip,
 * refreshAuthStripProgress, initInlineSignin(container), and wires sign-out.
 */
(function () {
    function getCsrfToken() {
        var match = document.cookie.match(/\bXSRF-TOKEN=([^;]*)/);
        return match ? decodeURIComponent(match[1].replace(/^\s+|\s+$/g, '')) : '';
    }
    window.getCsrfToken = getCsrfToken;

    function getAnonSessionId() {
        var key = 'ilga_anon_sid';
        try {
            if (typeof sessionStorage !== 'undefined' && sessionStorage.getItem(key)) {
                return sessionStorage.getItem(key);
            }
            if (typeof crypto !== 'undefined' && crypto.randomUUID) {
                var uuid = crypto.randomUUID();
                sessionStorage.setItem(key, uuid);
                return uuid;
            }
            if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
                var arr = new Uint8Array(16);
                crypto.getRandomValues(arr);
                var hex = '';
                for (var i = 0; i < arr.length; i++) hex += ('0' + arr[i].toString(16)).slice(-2);
                var sid = hex.substring(0, 32);
                sessionStorage.setItem(key, sid);
                return sid;
            }
            return null;
        } catch (e) {
            return null;
        }
    }
    window.getAnonSessionId = getAnonSessionId;

    window._ilgaUserEmail = null;
    window._pendingAuthEmail = '';

    function refreshAuthStripProgress() {
        /* Progress line ("Called X legislators and sent Y emails") removed from hero; kept as no-op for callers. */
    }
    window.refreshAuthStripProgress = refreshAuthStripProgress;

    function updateAuthStrip(signedIn, email) {
        var stripSignedOut = document.getElementById('auth-strip-signed-out');
        var stripSignedIn = document.getElementById('auth-strip-signed-in');
        var stripEmail = document.getElementById('auth-strip-email');
        if (!stripSignedOut || !stripSignedIn) return;
        if (signedIn && email) {
            window._ilgaUserEmail = email;
            stripSignedOut.hidden = true;
            stripSignedIn.hidden = false;
            if (stripEmail) stripEmail.textContent = email;
            stripSignedIn.classList.remove('auth-strip-signed-in--visible');
            requestAnimationFrame(function () {
                stripSignedIn.classList.add('auth-strip-signed-in--visible');
            });
            refreshAuthStripProgress();
        } else {
            window._ilgaUserEmail = null;
            stripSignedOut.hidden = false;
            stripSignedIn.hidden = true;
            stripSignedIn.classList.remove('auth-strip-signed-in--visible');
        }
    }
    window.updateAuthStrip = updateAuthStrip;

    (function initAuthStrip() {
        var stripSignout = document.getElementById('auth-strip-signout');
        if (stripSignout) {
            stripSignout.onclick = function () {
                var signedInEl = document.getElementById('auth-strip-signed-in');
                if (signedInEl && !signedInEl.hidden) {
                    signedInEl.classList.remove('auth-strip-signed-in--visible');
                    function finishSignout() {
                        signedInEl.removeEventListener('transitionend', finishSignout);
                        window._ilgaUserEmail = null;
                        updateAuthStrip(false);
                    }
                    signedInEl.addEventListener('transitionend', finishSignout, { once: true });
                    setTimeout(finishSignout, 560);
                } else {
                    window._ilgaUserEmail = null;
                    updateAuthStrip(false);
                }
                try { sessionStorage.removeItem('ilga_anon_sid'); } catch (e) {}
                fetch('/auth/logout', { method: 'POST', credentials: 'same-origin' })
                    .then(function () {
                        if (document.dispatchEvent) {
                            document.dispatchEvent(new CustomEvent('ilga:auth-change', { detail: { signedIn: false } }));
                        }
                    })
                    .catch(function () { });
            };
        }
        fetch('/auth/me', { credentials: 'same-origin' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.authenticated && data.email) {
                    window._ilgaUserEmail = data.email;
                    updateAuthStrip(true, data.email);
                } else {
                    window._ilgaUserEmail = null;
                    updateAuthStrip(false);
                }
            })
            .catch(function () { });
    })();

    /**
     * Wire inline sign-in (email → code → verify) inside container.
     * Container may be #auth-strip (hero: has .auth-strip-signin-btn and .hero-signin-inline)
     * or #poll-signin-wrap (poll fragment: only .hero-signin-inline).
     */
    function initInlineSignin(container) {
        if (!container) return;
        var inlineBlock = container.querySelector('.hero-signin-inline') || (container.classList && container.classList.contains('hero-signin-inline') ? container : null);
        if (!inlineBlock) return;

        var signinBtn = container.querySelector('.auth-strip-signin-btn') || (container.id === 'poll-signin-wrap' ? document.getElementById('poll-signin-btn') : null);
        var stripSignedOut = container.querySelector('#auth-strip-signed-out');
        var states = container.querySelectorAll('.hero-signin-state');
        var stateEmail = states[0];
        var stateCode = states[1];
        var emailInput = stateEmail && stateEmail.querySelector('.hero-signin-input');
        var requestBtn = stateEmail && stateEmail.querySelector('.hero-signin-btn');
        var codeHint = stateCode && stateCode.querySelector('.hero-signin-code-hint');
        var codeInput = stateCode && stateCode.querySelector('.hero-signin-code-input');
        var verifyBtn = stateCode && stateCode.querySelectorAll('.hero-signin-btn')[0];
        var codeError = stateCode && stateCode.querySelector('.hero-signin-error');
        var resendBtn = stateCode && stateCode.querySelector('.hero-signin-resend');
        var cancelBtn = container.querySelector('.hero-signin-cancel');

        var _inlineCloseTimer = null;

        function showInline(show) {
            if (!inlineBlock) return;
            if (show) {
                if (_inlineCloseTimer) { clearTimeout(_inlineCloseTimer); _inlineCloseTimer = null; }
                inlineBlock.hidden = false;
                inlineBlock.classList.remove('hero-signin-inline--open');
                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        inlineBlock.classList.add('hero-signin-inline--open');
                    });
                });
            } else {
                if (_inlineCloseTimer) clearTimeout(_inlineCloseTimer);
                inlineBlock.classList.remove('hero-signin-inline--open');
                _inlineCloseTimer = setTimeout(function () {
                    _inlineCloseTimer = null;
                    inlineBlock.hidden = true;
                }, 560);
            }
        }

        function showState(which) {
            if (stateEmail) {
                stateEmail.hidden = (which !== 'email');
                stateEmail.classList.toggle('hero-signin-state-visible', which === 'email');
            }
            if (stateCode) {
                stateCode.hidden = (which !== 'code');
                if (which === 'code') {
                    requestAnimationFrame(function () {
                        stateCode.classList.add('hero-signin-state-visible');
                    });
                } else {
                    stateCode.classList.remove('hero-signin-state-visible');
                }
            }
        }

        function closeInline() {
            if (_inlineCloseTimer) { clearTimeout(_inlineCloseTimer); _inlineCloseTimer = null; }
            if (inlineBlock) {
                inlineBlock.hidden = true;
                inlineBlock.classList.remove('hero-signin-inline--open');
            }
            if (stripSignedOut) stripSignedOut.hidden = false;
            showState('email');
            if (emailInput) emailInput.value = '';
            if (codeInput) codeInput.value = '';
            if (codeError) { codeError.textContent = ''; codeError.hidden = true; }
        }

        if (signinBtn) {
            signinBtn.onclick = function () {
                if (stripSignedOut) stripSignedOut.hidden = true;
                showInline(true);
                showState('email');
                requestAnimationFrame(function () {
                    if (stateEmail) stateEmail.classList.add('hero-signin-state-visible');
                });
                if (emailInput) { emailInput.value = ''; emailInput.focus(); }
                if (requestBtn) requestBtn.disabled = true;
                if (codeError) { codeError.hidden = true; codeError.textContent = ''; }
            };
        } else {
            showInline(true);
            showState('email');
            if (stateEmail) stateEmail.classList.add('hero-signin-state-visible');
        }

        function isEmailLike(val) {
            var s = (val || '').trim();
            return s.length > 0 && s.indexOf('@') >= 1 && s.indexOf('@') < s.length - 1;
        }
        function setSendCodeEnabled() {
            if (requestBtn) requestBtn.disabled = !isEmailLike(emailInput && emailInput.value);
        }

        if (emailInput) {
            emailInput.addEventListener('input', setSendCodeEnabled);
            emailInput.addEventListener('change', setSendCodeEnabled);
            emailInput.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (requestBtn && !requestBtn.disabled) requestBtn.click();
                }
            });
        }

        if (cancelBtn) cancelBtn.onclick = closeInline;

        if (requestBtn) {
            requestBtn.onclick = function () {
                var email = (emailInput && emailInput.value || '').trim().toLowerCase();
                if (!email || email.indexOf('@') === -1) { if (emailInput) emailInput.focus(); return; }
                requestBtn.disabled = true;
                requestBtn.classList.add('hero-signin-btn--sending');
                requestBtn.textContent = 'Sending\u2026';
                var body = new FormData();
                body.append('email', email);
                body.append('csrf_token', getCsrfToken());
                fetch('/auth/request-code', { method: 'POST', body: body, credentials: 'same-origin' })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        requestBtn.classList.remove('hero-signin-btn--sending');
                        requestBtn.disabled = false;
                        requestBtn.textContent = 'Send code';
                        if (data.ok) {
                            window._pendingAuthEmail = email;
                            if (codeHint) {
                                var esc = (function (s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); })(email);
                                codeHint.innerHTML = 'Code sent to <strong>' + esc + '</strong>';
                            }
                            if (codeInput) { codeInput.value = ''; codeInput.focus(); }
                            if (codeError) codeError.hidden = true;
                            showState('code');
                        } else {
                            if (codeError) { codeError.textContent = data.error || 'Couldn\'t send code'; codeError.hidden = false; }
                        }
                    })
                    .catch(function () {
                        requestBtn.classList.remove('hero-signin-btn--sending');
                        requestBtn.disabled = false;
                        requestBtn.textContent = 'Send code';
                        if (codeError) { codeError.textContent = 'Network error \u2014 try again'; codeError.hidden = false; }
                    });
            };
        }

        var isPollContext = container.getAttribute && container.getAttribute('data-poll-signin-init');

        if (isPollContext) {
            container.addEventListener('click', function (e) {
                var target = e.target;
                if (target.closest && target.closest('.poll-signup-notyet')) {
                    var prompt = container.querySelector('.poll-signup-prompt');
                    if (prompt) {
                        prompt.hidden = true;
                        prompt.setAttribute('aria-hidden', 'true');
                    }
                    e.preventDefault();
                    return;
                }
                if (target.closest && target.closest('.poll-signup-yes')) {
                    var prompt = container.querySelector('.poll-signup-prompt');
                    if (!prompt) return;
                    var yesBtn = prompt.querySelector('.poll-signup-yes');
                    if (yesBtn) {
                        yesBtn.disabled = true;
                        yesBtn.textContent = 'Adding…';
                    }
                    fetch('/updates/subscribe', {
                        method: 'POST',
                        credentials: 'same-origin',
                        headers: { 'HX-Request': 'true' },
                    })
                        .then(function (r) { return r.text(); })
                        .then(function (html) {
                            prompt.innerHTML = html;
                            prompt.hidden = false;
                            prompt.setAttribute('aria-hidden', 'false');
                        })
                        .catch(function () {
                            if (yesBtn) {
                                yesBtn.disabled = false;
                                yesBtn.textContent = 'Yes';
                            }
                        });
                    e.preventDefault();
                }
            });
        }

        function doVerify() {
            var code = (codeInput && codeInput.value || '').trim();
            if (!code || code.length < 6) return;
            if (verifyBtn) { verifyBtn.disabled = true; verifyBtn.textContent = 'Verifying\u2026'; }
            if (codeError) codeError.hidden = true;
            var body = new FormData();
            body.append('email', window._pendingAuthEmail);
            body.append('code', code);
            body.append('csrf_token', getCsrfToken());
            var anonSid = getAnonSessionId();
            if (anonSid) body.append('anon_session_id', anonSid);
            if (isPollContext) body.append('from_poll', '1');
            fetch('/auth/verify-code', { method: 'POST', body: body, credentials: 'same-origin' })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (verifyBtn) { verifyBtn.disabled = false; verifyBtn.textContent = 'Confirm'; }
                    if (data.ok && data.email) {
                        window._ilgaUserEmail = data.email;
                        try { sessionStorage.removeItem('ilga_anon_sid'); } catch (e) {}
                        if (_inlineCloseTimer) { clearTimeout(_inlineCloseTimer); _inlineCloseTimer = null; }
                        if (inlineBlock) {
                            inlineBlock.hidden = true;
                            inlineBlock.classList.remove('hero-signin-inline--open');
                        }
                        showState('email');
                        if (emailInput) emailInput.value = '';
                        if (codeInput) codeInput.value = '';
                        if (codeError) { codeError.textContent = ''; codeError.hidden = true; }
                        updateAuthStrip(true, data.email);
                        if (document.dispatchEvent) {
                            document.dispatchEvent(new CustomEvent('ilga:auth-change', { detail: { signedIn: true } }));
                        }
                        if (isPollContext) {
                            var wrap = container.closest('.kei-poll-card');
                            if (wrap && wrap.id) {
                                var pollId = wrap.id.replace(/-wrap$/, '') || 'footer-kei-poll';
                                fetch('/updates/kei-poll-results?poll_id=' + encodeURIComponent(pollId), {
                                    credentials: 'same-origin',
                                    headers: { 'HX-Request': 'true' }
                                })
                                    .then(function (r) { return r.ok ? r.text() : Promise.reject(new Error('Not authenticated')); })
                                    .then(function (html) {
                                        wrap.innerHTML = html;
                                        if (window.htmx) window.htmx.process(wrap);
                                    })
                                    .catch(function () { });
                            }
                        }
                    } else {
                        if (codeError) { codeError.textContent = data.error || 'That code didn\'t work — try again or resend.'; codeError.hidden = false; }
                        if (codeInput) { codeInput.value = ''; codeInput.focus(); }
                    }
                })
                .catch(function () {
                    if (verifyBtn) { verifyBtn.disabled = false; verifyBtn.textContent = 'Confirm'; }
                    if (codeError) { codeError.textContent = 'Network error \u2014 try again'; codeError.hidden = false; }
                });
        }

        if (verifyBtn) verifyBtn.onclick = doVerify;

        if (codeInput) {
            codeInput.addEventListener('input', function () {
                this.value = this.value.replace(/\D/g, '').slice(0, 6);
                if (this.value.length === 6) doVerify();
            });
        }

        if (resendBtn) {
            resendBtn.onclick = function () {
                if (!window._pendingAuthEmail) return;
                resendBtn.disabled = true;
                resendBtn.textContent = 'Sending\u2026';
                if (codeError) codeError.hidden = true;
                var body = new FormData();
                body.append('email', window._pendingAuthEmail);
                body.append('csrf_token', getCsrfToken());
                fetch('/auth/request-code', { method: 'POST', body: body, credentials: 'same-origin' })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        resendBtn.disabled = false;
                        resendBtn.textContent = 'Resend code';
                        if (data.ok && codeHint) {
                            var escResend = (function (s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); })(window._pendingAuthEmail);
                            codeHint.innerHTML = 'New code sent to <strong>' + escResend + '</strong>';
                        }
                        if (codeInput) { codeInput.value = ''; codeInput.focus(); }
                    })
                    .catch(function () {
                        resendBtn.disabled = false;
                        resendBtn.textContent = 'Resend code';
                        if (codeError) { codeError.textContent = 'Network error \u2014 try again'; codeError.hidden = false; }
                    });
            };
        }
    }
    window.initInlineSignin = initInlineSignin;

    (function initHeroInlineSignin() {
        var authStrip = document.getElementById('auth-strip');
        if (authStrip) initInlineSignin(authStrip);
    })();
})();

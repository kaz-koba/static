import React, {
  useState, useEffect, useCallback,
} from 'react';

import { useTranslation } from 'next-i18next';
import { useRouter } from 'next/router';
import ReactCardFlip from 'react-card-flip';

import { apiv3Post } from '~/client/util/apiv3-client';
import { LoginErrorCode } from '~/interfaces/errors/login-error';
import { IErrorV3 } from '~/interfaces/errors/v3-error';
import { toArrayIfNot } from '~/utils/array-utils';

type LoginFormProps = {
  username?: string,
  name?: string,
  email?: string,
  isRegistrationEnabled: boolean,
  isEmailAuthenticationEnabled: boolean,
  registrationMode?: string,
  registrationWhiteList: string[],
  isPasswordResetEnabled: boolean,
  isLocalStrategySetup: boolean,
  isLdapStrategySetup: boolean,
  isLdapSetupFailed: boolean,
  objOfIsExternalAuthEnableds?: any,
  isMailerSetup?: boolean
}
export const LoginForm = (props: LoginFormProps): JSX.Element => {
  const { t } = useTranslation();
  const router = useRouter();

  const {
    isLocalStrategySetup, isLdapStrategySetup, isLdapSetupFailed, isPasswordResetEnabled, isRegistrationEnabled,
    isEmailAuthenticationEnabled, registrationMode, registrationWhiteList, isMailerSetup, objOfIsExternalAuthEnableds,
  } = props;
  const isLocalOrLdapStrategiesEnabled = isLocalStrategySetup || isLdapStrategySetup;
  const isSomeExternalAuthEnabled = Object.values(objOfIsExternalAuthEnableds).some(elem => elem);

  // states
  const [isRegistering, setIsRegistering] = useState(false);
  // For Login
  const [usernameForLogin, setUsernameForLogin] = useState('');
  const [passwordForLogin, setPasswordForLogin] = useState('');
  const [loginErrors, setLoginErrors] = useState<IErrorV3[]>([]);
  // For Register
  const [usernameForRegister, setUsernameForRegister] = useState('');
  const [nameForRegister, setNameForRegister] = useState('');
  const [emailForRegister, setEmailForRegister] = useState('');
  const [passwordForRegister, setPasswordForRegister] = useState('');
  const [registerErrors, setRegisterErrors] = useState<IErrorV3[]>([]);

  useEffect(() => {
    const { hash } = window.location;
    if (hash === '#register') {
      setIsRegistering(true);
    }
  }, []);

  // functions
  const handleLoginWithExternalAuth = useCallback((e) => {
    const auth = e.currentTarget.id;

    window.location.href = `/passport/${auth}`;
  }, []);

  const resetLoginErrors = useCallback(() => {
    if (loginErrors.length === 0) return;
    setLoginErrors([]);
  }, [loginErrors.length]);

  const handleLoginWithLocalSubmit = useCallback(async(e) => {
    e.preventDefault();
    resetLoginErrors();

    const loginForm = {
      username: usernameForLogin,
      password: passwordForLogin,
    };

    try {
      const res = await apiv3Post('/login', { loginForm });
      const { redirectTo } = res.data;
      router.push(redirectTo ?? '/');
    }
    catch (err) {
      const errs = toArrayIfNot(err);
      setLoginErrors(errs);
    }
    return;

  }, [passwordForLogin, resetLoginErrors, router, usernameForLogin]);

  // separate errors based on error code
  const separateErrorsBasedOnErrorCode = useCallback((errors: IErrorV3[]) => {
    const loginErrorListForDangerouslySetInnerHTML: IErrorV3[] = [];
    const loginErrorList: IErrorV3[] = [];

    errors.forEach((err) => {
      if (err.code === LoginErrorCode.PROVIDER_DUPLICATED_USERNAME_EXCEPTION) {
        loginErrorListForDangerouslySetInnerHTML.push(err);
      }
      else {
        loginErrorList.push(err);
      }
    });

    return [loginErrorListForDangerouslySetInnerHTML, loginErrorList];
  }, []);

  // wrap error elements which use dangerouslySetInnerHtml
  const generateDangerouslySetErrors = useCallback((errors: IErrorV3[]): JSX.Element => {
    if (errors == null || errors.length === 0) return <></>;
    return (
      <div className="alert alert-danger">
        {errors.map((err, index) => {
          return <small key={index} dangerouslySetInnerHTML={{ __html: t(err.message, err.args) }}></small>;
        })}
      </div>
    );
  }, [t]);

  // wrap error elements which do not use dangerouslySetInnerHtml
  const generateSafelySetErrors = useCallback((errors: IErrorV3[]): JSX.Element => {
    if (errors == null || errors.length === 0) return <></>;
    return (
      <ul className="alert alert-danger">
        {errors.map((err, index) => {
          return (
            <li key={index}>
              {t(err.message, err.args)}<br/>
            </li>);
        })}
      </ul>
    );
  }, [t]);

  const renderLocalOrLdapLoginForm = useCallback(() => {
    const { isLdapStrategySetup } = props;

    // separate login errors into two arrays based on error code
    const [loginErrorListForDangerouslySetInnerHTML, loginErrorList] = separateErrorsBasedOnErrorCode(loginErrors);
    // Generate login error elements using dangerouslySetInnerHTML
    const loginErrorElementWithDangerouslySetInnerHTML = generateDangerouslySetErrors(loginErrorListForDangerouslySetInnerHTML);
    // Generate login error elements using <ul>, <li>
    const loginErrorElement = generateSafelySetErrors(loginErrorList);

    return (
      <>
        {isLdapSetupFailed && (
          <div className="alert alert-warning small">
            <strong><i className="icon-fw icon-info"></i>{t('login.enabled_ldap_has_configuration_problem')}</strong><br/>
            <span dangerouslySetInnerHTML={{ __html: t('login.set_env_var_for_logs') }}></span>
          </div>
        )}
        {loginErrorElementWithDangerouslySetInnerHTML}
        {loginErrorElement}

        <form role="form" onSubmit={handleLoginWithLocalSubmit} id="login-form">
          <div className="input-group">
            <div className="input-group-prepend">
              <span className="input-group-text">
                <i className="icon-user"></i>
              </span>
            </div>
            <input type="text" className="form-control rounded-0" data-testid="tiUsernameForLogin" placeholder="Username or E-mail"
              onChange={(e) => { setUsernameForLogin(e.target.value) }} name="usernameForLogin" />
            {isLdapStrategySetup && (
              <div className="input-group-append">
                <small className="input-group-text text-success">
                  <i className="icon-fw icon-check"></i> LDAP
                </small>
              </div>
            )}
          </div>

          <div className="input-group">
            <div className="input-group-prepend">
              <span className="input-group-text">
                <i className="icon-lock"></i>
              </span>
            </div>
            <input type="password" className="form-control rounded-0" data-testid="tiPasswordForLogin" placeholder="Password"
              onChange={(e) => { setPasswordForLogin(e.target.value) }} name="passwordForLogin" />
          </div>

          <div className="input-group my-4">
            <button type="submit" id="login" className="btn btn-fill rounded-0 login mx-auto" data-testid="btnSubmitForLogin">
              <div className="eff"></div>
              <span className="btn-label">
                <i className="icon-login"></i>
              </span>
              <span className="btn-label-text">{t('Sign in')}</span>
            </button>
          </div>
        </form>
      </>
    );
  }, [generateDangerouslySetErrors, generateSafelySetErrors, handleLoginWithLocalSubmit,
      isLdapSetupFailed, loginErrors, props, separateErrorsBasedOnErrorCode, t]);

  const renderExternalAuthInput = useCallback((auth) => {
    const authIconNames = {
      google: 'google',
      github: 'github',
      facebook: 'facebook',
      twitter: 'twitter',
      oidc: 'openid',
      saml: 'key',
      basic: 'lock',
    };

    return (
      <div key={auth} className="col-6 my-2">
        <button type="button" className="btn btn-fill rounded-0" id={auth} onClick={handleLoginWithExternalAuth}>
          <div className="eff"></div>
          <span className="btn-label">
            <i className={`fa fa-${authIconNames[auth]}`}></i>
          </span>
          <span className="btn-label-text">{t('Sign in')}</span>
        </button>
        <div className="small text-right">by {auth} Account</div>
      </div>
    );
  }, [handleLoginWithExternalAuth, t]);

  const renderExternalAuthLoginForm = useCallback(() => {
    const { isLocalStrategySetup, isLdapStrategySetup, objOfIsExternalAuthEnableds } = props;
    const isExternalAuthCollapsible = isLocalStrategySetup || isLdapStrategySetup;
    const collapsibleClass = isExternalAuthCollapsible ? 'collapse collapse-external-auth' : '';

    return (
      <>
        <div className="grw-external-auth-form border-top border-bottom">
          <div id="external-auth" className={`external-auth ${collapsibleClass}`}>
            <div className="row mt-2">
              {Object.keys(objOfIsExternalAuthEnableds).map((auth) => {
                if (!objOfIsExternalAuthEnableds[auth]) {
                  return;
                }
                return renderExternalAuthInput(auth);
              })}
            </div>
          </div>
        </div>
        <div className="text-center">
          <button
            type="button"
            className="btn btn-secondary btn-external-auth-tab btn-sm rounded-0 mb-3"
            data-toggle={isExternalAuthCollapsible ? 'collapse' : ''}
            data-target="#external-auth"
            aria-expanded="false"
            aria-controls="external-auth"
          >
            External Auth
          </button>
        </div>
      </>
    );
  }, [props, renderExternalAuthInput]);

  const handleRegisterFormSubmit = useCallback(async(e, requestPath) => {
    e.preventDefault();

    const registerForm = {
      username: usernameForRegister,
      name: nameForRegister,
      email: emailForRegister,
      password: passwordForRegister,
    };
    try {
      const res = await apiv3Post(requestPath, { registerForm });
      const { redirectTo } = res.data;
      router.push(redirectTo);
    }
    catch (err) {
      // Execute if error exists
      if (err != null || err.length > 0) {
        setRegisterErrors(err);
      }
    }
    return;
  }, [emailForRegister, nameForRegister, passwordForRegister, router, usernameForRegister]);

  const resetRegisterErrors = useCallback(() => {
    if (registerErrors.length === 0) return;
    setRegisterErrors([]);
  }, [registerErrors.length]);

  const switchForm = useCallback(() => {
    setIsRegistering(!isRegistering);
    resetLoginErrors();
    resetRegisterErrors();
  }, [isRegistering, resetLoginErrors, resetRegisterErrors]);

  const renderRegisterForm = useCallback(() => {
    let registerAction = '/register';

    let submitText = t('Sign up');
    if (isEmailAuthenticationEnabled) {
      registerAction = '/user-activation/register';
      submitText = t('page_register.send_email');
    }

    return (
      <React.Fragment>
        {registrationMode === 'Restricted' && (
          <p className="alert alert-warning">
            {t('page_register.notice.restricted')}
            <br />
            {t('page_register.notice.restricted_defail')}
          </p>
        )}
        { (!isMailerSetup && isEmailAuthenticationEnabled) && (
          <p className="alert alert-danger">
            <span>{t('security_settings.Local.please_enable_mailer')}</span>
          </p>
        )}

        {
          registerErrors != null && registerErrors.length > 0 && (
            <p className="alert alert-danger">
              {registerErrors.map((err, index) => {
                return (
                  <span key={index}>
                    {t(err.message)}<br/>
                  </span>
                );
              })}
            </p>
          )
        }

        <form role="form" onSubmit={e => handleRegisterFormSubmit(e, registerAction) } id="register-form">

          {!isEmailAuthenticationEnabled && (
            <div>
              <div className="input-group" id="input-group-username">
                <div className="input-group-prepend">
                  <span className="input-group-text">
                    <i className="icon-user"></i>
                  </span>
                </div>
                {/* username */}
                <input
                  type="text"
                  className="form-control rounded-0"
                  onChange={(e) => { setUsernameForRegister(e.target.value) }}
                  placeholder={t('User ID')}
                  name="username"
                  defaultValue={props.username}
                  required
                />
              </div>
              <p className="form-text text-danger">
                <span id="help-block-username"></span>
              </p>
              <div className="input-group">
                <div className="input-group-prepend">
                  <span className="input-group-text">
                    <i className="icon-tag"></i>
                  </span>
                </div>
                {/* name */}
                <input type="text"
                  className="form-control rounded-0"
                  onChange={(e) => { setNameForRegister(e.target.value) }}
                  placeholder={t('Name')}
                  name="name"
                  defaultValue={props.name}
                  required />
              </div>
            </div>
          )}

          <div className="input-group">
            <div className="input-group-prepend">
              <span className="input-group-text">
                <i className="icon-envelope"></i>
              </span>
            </div>
            {/* email */}
            <input type="email"
              className="form-control rounded-0"
              onChange={(e) => { setEmailForRegister(e.target.value) }}
              placeholder={t('Email')}
              name="email"
              defaultValue={props.email}
              required
            />
          </div>

          {registrationWhiteList.length > 0 && (
            <>
              <p className="form-text">{t('page_register.form_help.email')}</p>
              <ul>
                {registrationWhiteList.map((elem) => {
                  return (
                    <li key={elem}>
                      <code>{elem}</code>
                    </li>
                  );
                })}
              </ul>
            </>
          )}

          {!isEmailAuthenticationEnabled && (
            <div>
              <div className="input-group">
                <div className="input-group-prepend">
                  <span className="input-group-text">
                    <i className="icon-lock"></i>
                  </span>
                </div>
                {/* Password */}
                <input type="password"
                  className="form-control rounded-0"
                  onChange={(e) => { setPasswordForRegister(e.target.value) }}
                  placeholder={t('Password')}
                  name="password"
                  required />
              </div>
            </div>
          )}

          {/* Sign up button (submit) */}
          <div className="input-group justify-content-center my-4">
            <button
              className="btn btn-fill rounded-0"
              id="register"
              disabled={(!isMailerSetup && isEmailAuthenticationEnabled)}
            >
              <div className="eff"></div>
              <span className="btn-label">
                <i className="icon-user-follow"></i>
              </span>
              <span className="btn-label-text">{submitText}</span>
            </button>
          </div>
        </form>

        <div className="border-bottom"></div>

        <div className="row">
          <div className="text-right col-12 mt-2 py-2">
            <a href="#login" id="login" className="link-switch" onClick={switchForm}>
              <i className="icon-fw icon-login"></i>
              {t('Sign in is here')}
            </a>
          </div>
        </div>
      </React.Fragment>
    );
  }, [handleRegisterFormSubmit, isEmailAuthenticationEnabled, isMailerSetup,
      props.email, props.name, props.username,
      registerErrors, registrationMode, registrationWhiteList, switchForm, t]);

  return (
    <div className="noLogin-dialog mx-auto" id="noLogin-dialog">
      <div className="row mx-0">
        <div className="col-12">
          <ReactCardFlip isFlipped={isRegistering} flipDirection="horizontal" cardZIndex="3">
            <div className="front">
              {isLocalOrLdapStrategiesEnabled && renderLocalOrLdapLoginForm()}
              {isSomeExternalAuthEnabled && renderExternalAuthLoginForm()}
              {isLocalOrLdapStrategiesEnabled && isPasswordResetEnabled && (
                <div className="text-right mb-2">
                  <a href="/forgot-password" className="d-block link-switch">
                    <i className="icon-key"></i> {t('forgot_password.forgot_password')}
                  </a>
                </div>
              )}
              {/* Sign up link */}
              {isRegistrationEnabled && (
                <div className="text-right mb-2">
                  <a href="#register" id="register" className="link-switch" onClick={switchForm}>
                    <i className="ti ti-check-box"></i> {t('Sign up is here')}
                  </a>
                </div>
              )}
            </div>
            <div className="back">
              {/* Register form for /login#register */}
              {isRegistrationEnabled && renderRegisterForm()}
            </div>
          </ReactCardFlip>
        </div>
      </div>
      <a href="https://growi.org" className="link-growi-org pl-3">
        <span className="growi">GROWI</span>.<span className="org">ORG</span>
      </a>
    </div>
  );

};

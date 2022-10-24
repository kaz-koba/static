import loggerFactory from '~/utils/logger';

const logger = loggerFactory('growi:middleware:admin-required');

module.exports = (crowi, fallback = null) => {

  return function(req, res, next) {

    if (req.user != null && (req.user instanceof Object) && '_id' in req.user) {
      if (req.user.admin) {
        return next();
      }

      logger.warn('This user is not admin.');

      if (fallback != null) {
        return fallback(req, res, next);
      }
      return res.redirect('/');
    }

    logger.warn('This user has not logged in.');

    if (fallback != null) {
      return fallback(req, res, next);
    }
    return res.redirect('/login');
  };

};

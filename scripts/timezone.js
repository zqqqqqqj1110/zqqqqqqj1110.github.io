/* global hexo */
'use strict';

// Post URLs (/:year/:month/:day/) are formatted in the build machine's time zone,
// which is UTC on GitHub Actions. Pin it to the site's time zone so a post keeps
// the same URL no matter where the site is built.
process.env.TZ = hexo.config.timezone || 'Asia/Shanghai';

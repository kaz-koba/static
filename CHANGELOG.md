# Changelog

## [Unreleased](https://github.com/weseek/growi/compare/v5.1.5...HEAD)

*Please do not manually update this file. We've automated the process.*

## [v5.1.5](https://github.com/weseek/growi/compare/v5.1.4...v5.1.5) - 2022-10-04

### 💎 Features

- feat: Add option to not use crop modal on brand logo upload (#6677) @Yohei-Shiina

### 🚀 Improvement

- imprv: Emoji picker performance for v5.x (#6689) @hakumizuki

### 🐛 Bug Fixes

- fix(auditlog): Attachment download is displayed even if the filter is unchecked (#6688) @miya
- fix: firstName and lastName japanese translations in SAML  (#6631) @kaoritokashiki

## [v5.1.4](https://github.com/weseek/growi/compare/v5.1.3...v5.1.4) - 2022-09-12

### 💎 Features

- feat:  Truncate long path when recent changes is in S size (#6263) @mudana-grune
- feat: In-app notifications when removing descendants of subscribed pages (#6192) @Shunm634-source
- feat: Not increment ordered list number in CodeMirror (#6462) @mudana-grune

### 🚀 Improvement

- imprv: Added page URL to mail subject (#6554) @hakumizuki

### 🐛 Bug Fixes

- fix: Cannot update user group without parent (#6530) @kaoritokashiki
- fix: Make PageTree input not draggable when editting (#6525) @hakumizuki
- fix: Pagetree input hit enter (#6526) @hakumizuki
- fix: Disallow retrieval of revision data that does not match the page (#6537) @miya

## [v5.1.3](https://github.com/weseek/growi/compare/v5.1.2...v5.1.3) - 2022-08-28

### 💎 Features

- feat(auditlog): Copy URL of the table (#6421) @miya

### 🚀 Improvement

- imprv(auditlog): Activity paging UI (#6444) @miya
- imprv: Improvement behavior when click on drawio diagram. (#6486) @kaishuu0123

### 🐛 Bug Fixes

- fix: Label of alert when updating tags (#6478) @miya
- fix: Uploading image using shortcut key(ctrl+v) shows toastError (#6474) @Yohei-Shiina
- fix: Pager is not displayed (#6468) @miya

### 🧰 Maintenance

- support: Use vscode-stylelint (#6430) @yuki-takei

## [v5.1.2](https://github.com/weseek/growi/compare/v5.1.1...v5.1.2) - 2022-08-03

### 💎 Features

- feat: Make content width of each page configurable (#6107) @mudana-grune

### 🚀 Improvement

- imprv(auditlog): Clear and reload button (#6398) @miya
- imprv(auditlog): Date Range Picker  (#6395) @miya

### 🐛 Bug Fixes

- fix: MathJax rendering (#6396) @yuki-takei

### 🧰 Maintenance

- support: Make Editor component Functional Component and TypeScript (#6374) @yukendev

## [v5.1.1](https://github.com/weseek/growi/compare/v5.1.0...v5.1.1) - 2022-08-01

### 💎 Features

- feat: Users can set users per ip from env var at API Rate Limit  (#6379) @yukendev
- feat: Show user picture in Audit Log (#6342) @miya
- feat: Reset search criteria button (#6327) @miya

### 🚀 Improvement

- imprv(auditlog): Display number of actions that can be saved (#6353) @miya
- imprv(auditlog): Include delete-related actions in small group (#6351) @miya

### 🐛 Bug Fixes

- fix: Default markdown linker with relative path does not respect the current page path (v5.1.0) (#6378) @yuki-takei
- fix: Recover page path operation (#6368) @hakumizuki
- fix: Migration script for inserting NamedQuery (#6364) @yuki-takei
- fix: "Error: cannnot get grant label" occured with lsx (#6348) @yukendev

## [v5.1.0](https://github.com/weseek/growi/compare/v5.0.11...v5.1.0) - 2022-07-21

### 💎 Features

- feat: Custom brand logo image (#5709) @mudana-grune
- feat: Rate Limit by rate-limit-flexible (#6053) @yukendev
- feat: Audit Log (#5915) @miya

### 🚀 Improvement

- imprv: Prevent XSS with React (#6274) @yuki-takei
- imprv: Reflect tmp tag data (#6124) @kaoritokashiki
- imprv: Update subscribe button icon on Navbar (#6213) @jam411
- imprv: Event emittion by socket.io is triggered only when ES reindexing (#6077) @hirokei-camel

### 🐛 Bug Fixes

- fix: Drawio rendering (#6275) @hakumizuki
- fix: Blink section header on init (#6249) @yuki-takei
- fix: Error when trying login with an email that contains plus sign (#6232) @miya
- fix: Use APIv3 for api get check_username (#6226) @kaoritokashiki
- fix: Slack integration connection test (#6201) @yukendev
- fix: Not found page for `/${ObjectId like string}` path (#6208) @yuki-takei

### 🧰 Maintenance

- support: Refactor PageInfo types (#6283) @yuki-takei
- support: Refactor growi renderer using hooks 2 (#6237) @yuki-takei
- support: Refactor growi renderer using hooks (#6223) @hakumizuki
- imprv: Omit Personal Container (#6182) @kaoritokashiki

## [v5.0.11](https://github.com/weseek/growi/compare/v5.0.10...v5.0.11) - 2022-07-05

### 💎 Features

- feat: Integrate recount descendant count after paths fix (#6170) @Yohei-Shiina

### 🚀 Improvement

- imprv: Redirect when the anchor is #password (#6144) @Kami-jo

### 🐛 Bug Fixes

- fix: User registration page is not redirected after tmp login (#6197) @kaoritokashiki
- fix: Empty trash doesn't work (#6168) @yukendev

### 🧰 Maintenance

- support: Ease rate limit temporary (#6191) @yuki-takei
- support: Omit page history container and page revision comparer container (#6185) @yukendev

## [v5.0.10](https://github.com/weseek/growi/compare/v5.0.9...v5.0.10) - 2022-06-27

### 💎 Features

- feat: Sidebar default mode settings (#6111) @yukendev
- feat: Get GCS instance that uses Application Default Credentials for v5 (#6051) @Yohei-Shiina
- feat: Resume rename on server boot (#5862)(#6014) @Yohei-Shiina
- feat: Show page item control menu on empty page (#6070)(#6103) @Yohei-Shiina

### 🚀 Improvement

- imprv: Show page control on subnavigation at existing empty page  (#5638) @Yohei-Shiina
- imprv: Remove toc and page authors in empty page (#5661) @Yohei-Shiina
- imprv: SWRize apiGet /tag.search (#6062) @kaoritokashiki

### 🐛 Bug Fixes

- fix: Scrolling preview (#6148) @yuki-takei
- fix: Show page history comparation modal on init (#6072) @hirokei-camel
- fix: Ensure backword compatibility for ES6 when using max_analyzed_offset (#6121) @hakumizuki
- fix: Set max_analyzed_offset to elasticsearch querying options (#6115) @hakumizuki
- fix: Revision err when updating tags (#6073) @kaoritokashiki
- fix: Support 3 types of syntax for OpenID Connect Issuer Host (#6061) @mudana-grune

### 🧰 Maintenance

- support: Omit comment container (#6147) @yuki-takei
- support: Upgrade typescript to ^4.6.0 (#6082) @hakumizuki

## [v5.0.9](https://github.com/weseek/growi/compare/v5.0.8...v5.0.9) - 2022-06-13

### 🚀 Improvement

- imprv: Render MathJax in Preview tab of comment (#6025) @yuki-takei
- imprv: Exception handling for user authentication (#6019) @kaoritokashiki
- imprv: Sidebar background color on light theme and add shadow on dark theme (#6012) @shukmos
- imprv: Limit display of notification paths (#5991) @jam411

### 🐛 Bug Fixes

- fix: Getting page API is broken (#6023) @yuki-takei
- fix: MathJax does not working (#6020) @yuki-takei

## [v5.0.8](https://github.com/weseek/growi/compare/v5.0.7...v5.0.8) - 2022-06-07

### 🚀 Improvement

- imprv: Fix subnavigation spacing (#5995) @yuki-takei
- imprv: Set Content-Length header in response of attachment (#5972) @hiroki-hgs
- imprv: Fix sidebar tag layout (#5984) @jam411
- imprv: PageStatusAlert labels when data is outdated (#5961) @yuki-takei
- imprv: Delete NotFoundAlert from not found page (#5919) @Shunm634-source

### 🐛 Bug Fixes

- fix: Too many footstamps icons are shown by lsx output 3 (#6000) @yuki-takei
- fix: Adjust PageItemControl alignment (#5994) @yuki-takei
- fix: CodeMirror placeholder color (#5993) @yuki-takei
- fix: Chinese notation is broken on create new page modal (#5973) @jam411
- fix: Document timestamps does not updated (#5979) @yuki-takei
- fix: Slack channels are not automatically filled after setting up user trigger notification on v5.0.x (#5911) @kaoritokashiki
- fix: Login required when viewing sharelink page (#5959) @yuki-takei
- fix: Editor scroll sync by Preview scrolling does not work (#5949) @yuki-takei

### 🧰 Maintenance

- support: Enable garbage collection at runtime with expose-gc package (#5986) @yuki-takei
- support: Upgrade aws-sdk to v3 (#5863) @mudana-grune

## [v4.5.22](https://github.com/weseek/growi/compare/v4.5.21...v4.5.22) - 2022-06-07

### 🐛 Bug Fixes

- fix: Fixed the bug of auto-filling unintended values into the Email field of the User settings (#5885) @Shunm634-source
- fix: google-oauth2 Automatically bind external accounts does not work (#5891) @kaoritokashiki
- fix: Slack channels are not automatically filled after setting up user trigger notification (#5976) @kaoritokashiki

### 🧰 Maintenance

- support: Enable garbage collection at runtime with expose-gc package (#5998) @kaoritokashiki

## [v5.0.7](https://github.com/weseek/growi/compare/v5.0.6...v5.0.7) - 2022-05-30

### 💎 Features

- feat: Set the min length of passwords by environment variable (#5899) @Shunm634-source
- feat: API to find username (#5907) @miya

### 🐛 Bug Fixes

- fix: Page is not rendered for guest (#5930) @yuki-takei
- fix: Server error due to the canDeleteLogic method (#5927) @hakumizuki
- fix: Show pagename on toastr when page deleted (#5772) @hirokei-camel
- fix: Search result screen is broken under content 100% setting (#5917) @jam411

## [v5.0.6](https://github.com/weseek/growi/compare/v5.0.5...v5.0.6) - 2022-05-27

### 💎 Features

- feat: Emoji - replace emojione to emojimart (#5668) @kaoritokashiki
- feat: Show username suggestion for mention in comment (#5856) @mudana-grune
- feat: Send in-app notification when containing username mention in comment  (#5906) @mudana-grune
- feat: Customize menu in navbar for guest user (#5858) @yukendev
- feat: Admin only page convert by path (#5902) @hakumizuki
- feat: Fix grant alert (#5903) @hakumizuki

### 🚀 Improvement

- imprv: Automatic login after registration (#5860) @hiroki-hgs
- imprv: Add tooltip to SubNavButtons (#5887) @miya
- imprv: Mixin of argument-of-override-list-group-item-for-pagetree for dark theme (#5904) @shukmos
- imprv: Move code to the appropriate place for fix browser auto-complete email wiith username (#5892) @Yohei-Shiina
- imprv: Initial rendering when opening Custom Sidebar (#5880) @Kami-jo
- imprv: Add contributors to staff credit (#5841) @hiroki-hgs

### 🐛 Bug Fixes

- fix: Can not toggle textlint function on v5.0.x (#5854) @kaoritokashiki
- fix(google-oauth2): Automatically bind external accounts  does not work on v5.0.x (#5886) @kaoritokashiki

## [v4.5.21](https://github.com/weseek/growi/compare/v4.5.20...v4.5.21) - 2022-05-23

### 🐛 Bug Fixes

- fix: Can not toggle textlint function on v4.5.x (https://github.com/weseek/growi/pull/5855) @kaoritokashiki
- fix: Error on searching (https://github.com/weseek/growi/pull/5873) @miya

## [v5.0.5](https://github.com/weseek/growi/compare/v5.0.4...v5.0.5) - 2022-05-16

### 💎 Features

- feat: Empty trash button in trash page (#5816) @yukendev

### 🚀 Improvement

- imprv: Count badge colors (#5835) @shukmos
- imprv: List group background colors on PageTree (#5812) @shukmos
- imprv: Page path auto complete function for page rename modal (#5805) @kaoritokashiki
- imprv: Show toastr when converting is completed on Private Legacy Page (#5810) @yukendev
- imprv: Create parent pages as needed by path that includes slash (#5809) @kaoritokashiki

### 🐛 Bug Fixes

- fix: Change the execution user of the official docker image to root (#5846) @yuki-takei
- fix: Display admin link only with logged in (#5799) @hirokei-camel
- fix: Error when renaming (#5793) @miya

### 🧰 Maintenance

- support: Typescriptize tag model (#5778) @kaoritokashiki

## [v4.5.20](https://github.com/weseek/growi/compare/v4.5.19...v4.5.20) - 2022-05-12

### 🐛 Bug Fixes

- fix: Guest user cannot access share link pages (#5819) @kaoritokashiki

## [v5.0.4](https://github.com/weseek/growi/compare/v5.0.3...v5.0.4) - 2022-04-28

### 💎 Features

- feat: Private legacy pages convert by path (#5787) @hakumizuki
- feat: Generate activity when page is created (#5765) @miya
- feat: Private legacy pages convert by path API (#5760) @hakumizuki
- feat:  Create notification when page is reverted (#5756) @miya
- feat: Create notification when page is duplicated (#5749) @miya
- feat: Add count badge to Page List button and Comment button (#5740) @yukendev
- feat: Infinite scroll for Recent Changes in Sidebar (#5647) @mudana-grune

### 🚀 Improvement

- imprv: Change GET method to POST for logout operation (#5751) @kaoritokashiki
- imprv: Redesign tags (#5730) @miya
- imprv: i18n for already_exists error in PutBackPageModal (#5747) @kaoritokashiki

### 🐛 Bug Fixes

- fix: Default markdown linker with relative path does not respect the current page path (#5788) @yuki-takei
- fix: Include any public pages as applicable ancestors (#5786) @hakumizuki
- fix: Not create unnecessary empty pages when ancestors are public (#5774) @hakumizuki
- fix: Too many footstamps icons are shown by lsx output 2 (#5763) @yuki-takei
- fix:  footstamp-icon size (#5759) @kaoritokashiki

## [v4.5.19](https://github.com/weseek/growi/compare/v4.5.18...v4.5.19) - 2022-04-28

### 🐛 Bug Fixes

- fix: Swiping to previous/next page for Mac users (4.5.x) (#5758) @hirokei-camel
- fix: Get attachment list api without "page" parameter returns 500 response (#5726) @miya

## [v5.0.3](https://github.com/weseek/growi/compare/v5.0.2...v5.0.3) - 2022-04-21

### 💎 Features

- feat: Search on private legacy pages (#5723) @hakumizuki

### 🚀 Improvement

- imprv: Dark theme color optimization (#5737) @shukmos
- imprv: Change the order of menu items (#5722) @miya

### 🐛 Bug Fixes

- fix: Get attachment list api without "page" parameter returns 500 response (#5726) @miya
- fix: New user notification email is also sent TO: deleted_at_<epoch_time>@deleted (#5735) @yuki-takei
- fix: Too many footstamps icons are shown by lsx output (#5727) @yuki-takei

## [v5.0.2](https://github.com/weseek/growi/compare/v5.0.1...v5.0.2) - 2022-04-15

### 🐛 Bug Fixes

- fix: Edit button to open built-in editor does not work when HackMD is disabled (#5719) @yuki-takei
- fix: Share link list occures error when related page is not found (#5718) @yuki-takei

## [v5.0.1](https://github.com/weseek/growi/compare/v5.0.0...v5.0.1) - 2022-04-15

### 💎 Features

- feat: Input Slack member ID (#5412) @mudana-grune
- feat: Remove child group from parent group (#5600) @miya

### 🚀 Improvement

- imprv: Add spinner to tag sidebar (#5700) @miya
- imprv: Adjust pagelist and comment position (#5682) @Yohei-Shiina
- imprv: Adjust layout for PageTree Descendant Count (#5666) @miya
- imprv: adjust spaces in page item control and subnav btn (#5655) @Yohei-Shiina
- imprv: Clickable area of PageListItemL (#5665) @yuki-takei
- imprv: Add an expiration date for the link in the email (#5660) @miya
- imprv: remove min-width from search-sort-option-btn (#5656) @kaoritokashiki

### 🐛 Bug Fixes

- fix: Correction of expiredAt attached to email (#5715) @miya
- fix: Normalize parent so it does not include siblings (#5678) @hakumizuki
- fix: Prevent auto completing email with username stored by browser in /me page (#5702) @Yohei-Shiina
- fix: Do not include granted users if change page permission restricted (#5693) @miya
- fix: Do not include in search results if the page grant is restricted (#5691) @miya
- fix: Password reset gives error update password failed when submitting a new password (#5685) @kaoritokashiki
- fix: Cannot register new users (#5683) @kaoritokashiki
- fix: Sync change of count for both like and bookmark in search page (#5667) @Yohei-Shiina
- imprv: Adjust layout for PageTree Descendant Count (#5666) @miya
- fix: HackMD disabled tooltip on mobile (#5658) @yuki-takei
- fix: One Time Token is not available (#5654) @miya
- fix: Page items disappear when dnd (#5651) @miya

### 🧰 Maintenance

- ci(deps): bump anothrNick/github-tag-action from 1.36.0 to 1.38.0 (#5271) @dependabot
- ci(deps): bump amannn/action-semantic-pull-request from 3.4.5 to 4.2.0 (#5627) @dependabot
- ci(deps): bump actions/upload-artifact from 2 to 3 (#5686) @dependabot
- ci(deps): bump actions/download-artifact from 2 to 3 (#5687) @dependabot
- support: Migration for setting sparce option to slack member id (#5694) @kaoritokashiki
- support: Update eslint-config-weseek (#5673) @yuki-takei

## [v4.5.18](https://github.com/weseek/growi/compare/v4.5.17...v4.5.18) - 2022-04-15

### 🐛 Bug Fixes

- fix: One Time Token is not available for v4.5.x (#5713) @miya
- fix: Prevent auto completing email with username stored by browser in /me page for v4.5.x (#5703) @Yohei-Shiina
- fix: Page view count stops at 15 (#5705) @miya

## [v4.5.17](https://github.com/weseek/growi/compare/v4.5.16...v4.5.17) - 2022-04-07

### 🐛 Bug Fixes

- fix: Elasticsearch doesn't work properly on production (#5676) @Yohei-Shiina

## [v4.5.16](https://github.com/weseek/growi/compare/v4.5.15...v4.5.16) - 2022-04-06

### 💎 Features

- feat: Support Elasticsearch 7 (#5613) @Yohei-Shiina

### 🐛 Bug Fixes

- fix: Domain whitelist is not respected (fix #5408) (#5488) @yuto-oweseek
- fix: Add tags to pages restricted by specified groups on View mode (for v4.5.x) (#5487) @yuto-oweseek

## [v5.0.0](https://github.com/weseek/growi/compare/v4.5.15...v5.0.0) - 2022-04-01

### 💎 Features

- feat: Support Elasticsearch 7 (#5080) @yuki-takei
- feat: Elasticsearch reindex on boot (#5149) @LuqmanHakim-Grune
- feat: PageTree and re-impl SearchResult with list group (#5286) @yuki-takei
- feat: Rename(Move) by Drag & Drop (#5292) @hakumizuki
- feat: Maintenance mode (#5486) @hakumizuki
- feat: Delete permission config (#5527) @hakumizuki

### 🚀 Improvement

- imprv: Show comments in search page result (#5645) @yuki-takei
- imprv: Add description for user addition (#5614) @hakumizuki
- imprv: Validate deletion settings (#5581) @hakumizuki

### 🐛 Bug Fixes

- fix: Swiping to previous/next page for Mac users (5.0.x) (#5491) @hakumizuki
- fix: Guest User Access Dropdown shows wrong value (#5643) @miya
- fix: Show full text on presentation mode (#5636) @hakumizuki
- fix: Displaying minimum length of password (#5630) @Yohei-Shiina
- fix: Domain whitelist is not respected (fix #5408) (#5470) @yuto-oweseek
- fix: Add tags to pages restricted by specified groups on View mode (#5457) @yuto-oweseek

### 🧰 Maintenance

- ci(deps-dev): bump plantuml-encoder from 1.2.5 to 1.4.0 (#5633) @dependabot
- ci(deps-dev): bump codemirror from 5.63.0 to 5.64.0 (#4777) @dependabot
- ci(deps): bump nanoid from 3.1.30 to 3.2.0 (#5142) @dependabot
- support: Upgrade openid client (#5185) @mudana-grune
- ci(deps): bump amannn/action-semantic-pull-request from 3.4.2 to 3.4.5 (#4559) @dependabot
- ci(deps): bump extend from 3.0.1 to 3.0.2 (#5222) @dependabot
- ci(deps-dev): bump jquery-ui from 1.12.1 to 1.13.0 (#4548) @dependabot
- ci(deps): bump actions/setup-node from 2 to 3 (#5437) @dependabot
- ci(deps): bump actions/checkout from 2 to 3 (#5462) @dependabot
- ci(deps): bump peter-evans/dockerhub-description from 2 to 3 (#5615) @dependabot
- ci(deps): bump actions/cache from 2 to 3 (#5584) @dependabot
- ci(deps-dev): bump reveal.js from 3.6.0 to 4.3.1 (#5603) @dependabot
- support: Update yarn git-hosted-info v2.8.8 to v2.8.9 (#5215) @LuqmanHakim-Grune
- support: dependabot trim-off-newlines (#5336) @mudana-grune
- support: dependabot @npmcli/git (#5337) @mudana-grune
- support: dependabot highlight.js (#5352) @mudana-grune
- support: dependabot extend (#5335) @mudana-grune
- support: dependabot ajv (#5333) @mudana-grune
- support: dependabot dot-drop (#5204) @LuqmanHakim-Grune
- support: update nanoid yarn.lock v3.1.30 to v3.2.0 (#5216) @LuqmanHakim-Grune
- support: update validator version (#5562) @LuqmanHakim-Grune

## [v4.5.15](https://github.com/weseek/growi/compare/v4.5.14...v4.5.15) - 2022-02-17

### 🚀 Improvement

- imprv: Hide forgot password when localstrategy is disabled (#5380) @yuki-takei

### 🐛 Bug Fixes

- fix: The condition to attempt to reconnect to Elasticsearch (#5344) @yuki-takei
- fix: Highlight-addons and drawio-viewer for view missing (#5376) @yuki-takei

### 🧰 Maintenance

- support:  modify docker-compose indent (#5322) @yuto-oweseek

## [v4.5.14](https://github.com/weseek/growi/compare/v4.5.13...v4.5.14) - 2022-02-10

### 💎 Features

- feat: OGP in public wiki (#5304) @yuto-oweseek

## [v4.5.13](https://github.com/weseek/growi/compare/v4.5.12...v4.5.13) - 2022-02-08

### 🐛 Bug Fixes

- fix: fix: Sidebar collapsing (#5283) @yuki-takei

## [v4.5.12](https://github.com/weseek/growi/compare/v4.5.11...v4.5.12) - 2022-02-01

### 🚀 Improvement

- imprv: Sidebar opening delay (for v4.5.x) (#5218) @yuki-takei

### 🐛 Bug Fixes

- fix: /_api/v3/page with pageId param occurs an 500 error (#5212) @yuki-takei
- fix: Resolving OIDC issure host (#5220) @yuki-takei

## [v4.5.11](https://github.com/weseek/growi/compare/v4.5.10...v4.5.11) - 2022-01-26

### 🐛 Bug Fixes

- fix: Internal server error occured when "Restrict complete deletion of pages" option's value is "Admin and author" (#5175 ) @yuki-takei

## [v4.5.10](https://github.com/weseek/growi/compare/v4.5.9...v4.5.10) - 2022-01-26

### 💎 Features

- feat: Automatic installation (#5141) @yuki-takei

### 🚀 Improvement

- imprv: Migrate like states to swr (#5137) @miya

### 🐛 Bug Fixes

- fix: 86631-cannot-reset-password-in-case-that-register-limitation-is-Closed (#5155) @kaoritokashiki

### 🧰 Maintenance

- support: VRT with Cypress (#5030) @yuki-takei

## [v4.5.9](https://github.com/weseek/growi/compare/v4.5.8...v4.5.9) - 2022-01-21

### 🚀 Improvement

- imprv: 79291 make password min length 8 charactors (#5116) @kaoritokashiki

### 🐛 Bug Fixes

- fix: OIDC reconnection bug fix (#5104) @mudana-grune
- fix: /_api/v3/page is broken and dump 500 error "get-page-failed TypeError: user.canDeleteCompletely is not a function" (#5103) @yuki-takei
- fix: Default completely deletion settings label mismatched against to actual (#5102) @yuki-takei
- fix: OIDC issuer host availability check (#5099) @mudana-grune

### 🧰 Maintenance

- support: Improve multistage build (#5090) @yuki-takei
- support: Omit node-re2 (#5089) @yuki-takei
- ci(deps-dev): bump swr from 1.0.1 to 1.1.2 (#5018) @dependabot

## [v4.5.8](https://github.com/weseek/growi/compare/v4.5.7...v4.5.8) - 2022-01-12

### 💎 Features

- feat: Display a list of bookmarked users (#5044) @miya

### 🐛 Bug Fixes

- fix: Built-in editor scroll position is reset after save (Introduced by v4.5.3) (#5074) @yuki-takei

### 🧰 Maintenance

- support: Bump y18n to v4.0.3 (#5071) @yuki-takei
- support: Omit prettier-stylelint (#5070) @yuki-takei
- support: Bump tar to 6.1.11 (#5069) @yuki-takei

## [v4.5.7](https://github.com/weseek/growi/compare/v4.5.6...v4.5.7) - 2022-01-11

### 🐛 Bug Fixes

- fix: Subnavigation sticking initialization (#5062) @yuki-takei
- fix: Built-in editor was broken (#5061) @yuki-takei

### 🧰 Maintenance

- support: Bump re2 to 1.17.2 (#5059) @yuki-takei

## [v4.5.6](https://github.com/weseek/growi/compare/v4.5.5...v4.5.6) - 2022-01-07

### 💎 Features

- feat: Resolve conflict with 3-way merge like editor (#5012) @yuto-oweseek

### 🚀 Improvement

- imprv: Subnavigation behavior (#5047) @yuki-takei
- imprv: Switching editor mode behavior (#5043) @yuki-takei

### 🐛 Bug Fixes

- Bug: Error: The specified instance couldn't register because same id has already been registered (#5031) by #5043 @yuki-takei

## [v4.5.5](https://github.com/weseek/growi/compare/v4.5.4...v4.5.5) - 2022-01-05

### 💎 Features

- feat: OIDC reconnection (#5016) @mudana-grune
- feat: In-App Notification (#4792) @kaoritokashiki

### 🚀 Improvement

- imprv: Improve tags functions (#5001) @yuto-oweseek
- imprv: Migrate editor container grant to SWR (#4957) @stevenfukase

### 🐛 Bug Fixes

- Bug: Error: The specified instance couldn't register because same id has already been registered (#5031) by 573216c @yuki-takei

### 🧰 Maintenance

- fix: dependabot alert trim-newlines (#4931) @mudana-grune
- fix: dependabot alert dot-prop (#4921) @mudana-grune
- ci(deps-dev): bump tsconfig-paths-webpack-plugin from 3.5.1 to 3.5.2 (#4852) @dependabot
- ci(deps): bump ua-parser-js from 0.7.17 to 0.7.31 (#4895) @dependabot
- support: dependabot alert ssri (#4973) @mudana-grune

## [v4.5.4](https://github.com/weseek/growi/compare/v4.5.3...v4.5.4) - 2021-12-23

### 💎 Features

- feat: Hotkey to focus to search (#5006) @yuki-takei

### 🚀 Improvement

- imprv: Omit magnifier icon from global SearchTypeahead (#5005) @yuki-takei
- imprv: Focus to input when resetting SearchTypeahead (#5003) @yuki-takei
- imprv: Make updatedAt SWR (#4954) @kaoritokashiki
- imprv: Make createdAt SWR (#4819) @kaoritokashiki
- imprv: Performance optimization for large drawio diagrams (#4221) @kaishuu0123

### 🐛 Bug Fixes

- fix: Sidebar height is a little large (#4988) @yuki-takei

### 🧰 Maintenance

- imprv: Focus to input when resetting SearchTypeahead (#5003) @yuki-takei
- support: Typescriptize search components (#4982) @yuki-takei
- fix:  dependabot alert object-path (#4964) @mudana-grune
- fix: dependabot alert axios (#4960) @mudana-grune
- fix: dependabot alert elliptic (#4959) @mudana-grune
- fix: dependabot alert acorn (#4951) @mudana-grune
- fix: dependabot alert is-svg (#4937) @mudana-grune
- fix: dependabot alert socket.io-parser (#4934) @mudana-grune
- fix: dependabot alert serialize-javascript (#4910) @mudana-grune
- fix: dependabot alert js-yaml (#4906) @mudana-grune
- ci(deps): bump ws from 7.5.1 to 8.3.0 (#4728) @dependabot
- support: omit growi-commons (#4938) @yuki-takei

## [v4.5.3](https://github.com/weseek/growi/compare/v4.5.2...v4.5.3) - 2021-12-17

### 💎 Features

- feat: user activation by email (#4862) @kaoritokashiki

### 🚀 Improvement

- imprv: Use SWR for isSlackEnabled (#4827) @stevenfukase
- imprv: Disable rubber band scroll for Mac & iOS users (#4834) @hakumizuki
- imprv: Omit atlaskit and implement sidebar only with original codes (#4598) @yuki-takei

### 🐛 Bug Fixes

- fix: GROWI Bot search command after transplanting search service from dev/5.0.x (#4916) @hakumizuki
- fix: Set min-height to sidebar scroll target (#4884) @yuki-takei

### 🧰 Maintenance

- support: fix dependabot alert for kind-of (#4891) @LuqmanHakim-Grune
- support: fix dependabot alert for ini (#4892) @LuqmanHakim-Grune
- support: fix and debug mixin-deep dependabot alert (#4867) @LuqmanHakim-Grune
- support: dependabot alert xmlhttprequest-ssl (#4878) @mudana-grune
- support: Transplant search service from dev/5.0.x (#4869) @hakumizuki
- support: dependabot alert set-value (#4864) @LuqmanHakim-Grune
- ci(deps): bump aws-sdk from 2.179.0 to 2.1044.0 (#4821) @dependabot

## [v4.5.2](https://github.com/weseek/growi/compare/v4.5.1...v4.5.2) - 2021-12-06

### 🐛 Bug Fixes

- fix: Added scope for unfurl (#4811) @hakumizuki

## [v4.5.1](https://github.com/weseek/growi/compare/v4.5.0...v4.5.1) - 2021-12-06

### 🐛 Bug Fixes

- fix: /admin/slack-integration page dump undefined error (#4806) @yuki-takei

## [v4.5.0](https://github.com/weseek/growi/compare/v4.4.13...v4.5.0) - 2021-12-06

### BREAKING CHANGES

- imprv: APIv3 payload (#4770) @LuqmanHakim-Grune

### 💎 Features

- feat: Slackbot unfurl (#4720) @hakumizuki

### 🚀 Improvement

- imprv: APIv3 payload (#4770) @LuqmanHakim-Grune
- imprv: upgrade passport from v0.4.x to v0.5.x (#4727) @mudana-grune
- imprv: Show site url in unfurl footer (#4755) @hakumizuki
- imprv: SWRize context (#4740) @hakumizuki
- imprv: Upgrade mongoose from 5.x to 6.x (#4659) @mudana-grune

### 🐛 Bug Fixes

- fix(slackbot-proxy): Support new API v3 data scheme (#4800) @yuki-takei
- fix(Slackbot): Slash commands response when sent from disabled channels (#4754) @stevenfukase

### 🧰 Maintenance

- ci(deps): bump detect-indent from 6.0.0 to 7.0.0 (#4635) @dependabot
- ci(deps): bump passport-saml from 2.2.0 to 3.2.0 (#4431) @dependabot

## [v4.4.13](https://github.com/weseek/growi/compare/v4.4.12...v4.4.13) - 2021-11-19

### 💎 Features

- feat: Including comments in full text search (#4703) @kaoritokashiki

### 🐛 Bug Fixes

- fix(slackbot): Interactions from private channels not working (#4688) @stevenfukase

## [v4.4.12](https://github.com/weseek/growi/compare/v4.4.11...v4.4.12) - 2021-11-15

### 🐛 Bug Fixes

- fix: Cannot use HackMD (#4667)

### 🧰 Maintenance

- ci(deps): Downgrade passport to 0.4.0 (#4669) @mudana-grune

## [v4.4.11](https://github.com/weseek/growi/compare/v4.4.10...v4.4.11) - 2021-11-12

### 🚀 Improvement

- imprv: SAML settings by DB (#4656) @yuki-takei

### 🐛 Bug Fixes

- fix: Unescape Attribute-based Login Control field value (#4651) @haruhikonyan
- fix: Slack Integration 'note' command causes expired_trigger_id error (#4629) @stevenfukase
- fix: Timeline was broken (#4639) @yuki-takei

### 🧰 Maintenance

- support: Bump mpath with mongoose (#4638) @yuki-takei
- ci(deps): bump passport-oauth2 from 1.4.0 to 1.6.1 (#4599) @dependabot
- ci(deps): bump passport from 0.4.0 to 0.5.0 (#4582) @dependabot
- ci(deps): bump axios from 0.21.1 to 0.24.0 (#4604) @dependabot
- ci(deps): bump tar from 4.4.13 to 4.4.19 (#4601) @dependabot

## [v4.4.10](https://github.com/weseek/growi/compare/v4.4.9...v4.4.10) - 2021-11-08

### 🚀 Improvement

- imprv: Sidebar content header style (#4526) @yuki-takei

### 🐛 Bug Fixes

- fix: /pages/info API is broken (#4602) @yuki-takei
- fix: blackboard theme location in theme list (#4506) @ayaka0417

### 🧰 Maintenance

- support: Use SWR (#4487) @yuki-takei
- support: Replaced PageList with SWR (#4498) @takayuki-t
- support: Improve devcontainer (#4510) @yuki-takei
- support: Update passport-ldpauth from ^2.0.0 to ^3.0.1 (#4578) @LuqmanHakim-Grune
- ci(deps): bump validator from 13.6.0 to 13.7.0 (#4588) @dependabot
- ci(deps-dev): bump stylelint from 13.2.0 to 14.0.1 (#4583) @dependabot
- Bump browserslist from 4.0.1 to 4.16.6 (#3776) @dependabot
- ci(deps-dev): bump colors from 1.2.5 to 1.4.0 (#4365) @dependabot
- ci(deps-dev): bump on-headers from 1.0.1 to 1.0.2 (#4366) @dependabot
- ci(deps-dev): bump jquery-ui from 1.12.1 to 1.13.0 (#4549) @dependabot
- docs(page): Add docs to /page/info api (#4531) @Mxchaeltrxn

## [v4.4.9](https://github.com/weseek/growi/compare/v4.4.8...v4.4.9) - 2021-10-18

### 💎 Features

- feat: blackboard theme (#4501) @ayaka0417
- feat: jade-green theme (#4500) @ayaka0417
- feat: fire-red theme (#4499) @ayaka0417
- feat: Add user list for like button (#4346) @Mxchaeltrxn

### 🚀 Improvement

- imprv: GROWI slackbot help message (#4488) @hakumizuki

### 🐛 Bug Fixes

- fix: Migration update-mail-transmission (#4482) @yuki-takei

### 🧰 Maintenance

- support: Localize Copy bug report button (#4436) @AbiFirmandhani-Grune

## [v4.4.8](https://github.com/weseek/growi/compare/v4.4.7...v4.4.8) - 2021-10-08

### 🚀 Improvement

- imprv: Permissions to operate comment (#4466) @yuki-takei
- imprv: Show modal when enabling Textlint (#4373) @stevenfukase
- imprv: Slackbot reaction to user (#4442) @yuki-takei

### 🐛 Bug Fixes

- fix: Redirected to apiv3 endpoint when guest mode is enabled (#4443) @stevenfukase
- fix: Unnecessary extra JSON.stringify for configurations for slackbot without proxy (#4467) @hakumizuki
- fix: Migration for slackbot configurations without proxy (#4465) @hakumizuki
- fix: Slackbot error/command handling (#4463) @hakumizuki
- fix(slackbot): Respond bad gateway error & improved help message (#4470) @hakumizuki
- fix(slackbot): Stop auto-join to channels with middlewarer (#4424) @yuki-takei

## [v4.4.7](https://github.com/weseek/growi/compare/v4.4.6...v4.4.7) - 2021-09-29

### 🚀 Improvement

- imprv: Slackbot search (#4420) @yuki-takei
- imprv: Omit textlint-rule-en-capitalization (#4403) @yuki-takei
- imprv: Apply terminus for graceful shutdown (#4398) @yuki-takei

### 🐛 Bug Fixes

- fix: A problem that GROWI server doesn't retrieve connection status from Official bot proxy (#4416) @yuki-takei
- fix: Dictionary path of kuromoji invalid when uploaded to server (#4381) @stevenfukase
- fix: Copy correct dotenv file for NO_CDN docker image (#4397) @yuki-takei
- fix: Stop using ts-node in production (#4411) @yuki-takei
- fix: SAML setting says 'setup is not yet complete' even if setup properly (#4390) @nakashimaki
- fix: SidebarSmall button does not keep selection on reload (#4389) @nakashimaki
- fix: Migrations for updating data for slackbot (#4406) @yuki-takei
- fix: Migrations do not run in production (#4395) @yuki-takei
- fix: Migration file for mongodb 3.6 compatibility (#4413) @hakumizuki
- fix(slackbot): Sync permission when data stored is not enough (#4417) @yuki-takei

### 🧰 Maintenance

- support: Install Git LFS when provisioning of devcontainer (#4405) @stevenfukase
- chore: Add .dockerignore (#4396) @yuki-takei

## [v4.4.6](https://github.com/weseek/growi/compare/v4.4.5...v4.4.6) - 2021-09-24

### 🚀 Improvement

- imprv: Slackbot response flow (#4296) @yuki-takei
- imprv(slackbot-proxy): Show version on the top page (#4342) @yuto-oweseek

### 🧰 Maintenance

- support(slackbot-proxy): Bump slackbot proxy version independentry (#4385) @yuki-takei

## [v4.4.5](https://github.com/weseek/growi/compare/v4.4.4...v4.4.5) - 2021-09-23

### 🐛 Bug Fixes

- fix: Revert #4347
- fix: ERROR: Cannot find module 'tslib' on v4.4.4 (#4368) @yuki-takei

### 🧰 Maintenance

- support: bump @promster/express and @promster/server (#4370) @yuki-takei
- support: Upgrade codemirror to 5.63.0 (#4364) @yuki-takei
- ci(deps-dev): bump codemirror from 5.48.4 to 5.58.2 (#4363) @dependabot

## [v4.4.4](https://github.com/weseek/growi/compare/v4.4.3...v4.4.4) (Discontinued) - 2021-09-22

### 💎 Features

- feat: Add Textlint support (#4228) @kaoritokashiki

### 🚀 Improvement

- imprv: Highlighting searching keyword (#4327) @yuki-takei

### 🐛 Bug Fixes

- fix: Backspace key on last line doesn't work in vim mode (#4347) @yuki-takei
- fix: Recent Created of home is empty (#4345) @yuki-takei
- fix: IME suggestion list obscures inputted text (#4335) @yuki-takei
- fix: Highlighting section header (#4326) @yuki-takei

### 🧰 Maintenance

- chore: Update passport-saml 2.2.0 (#4360) @LuqmanHakim-Grune
- ci(deps): bump http-errors from 1.6.2 to 1.8.0 (#4353) @dependabot
- ci(deps-dev): bump @tsed/json-mapper from 6.43.0 to 6.70.1 (#4352) @dependabot
- ci(deps): bump graceful-fs from 4.1.11 to 4.2.8 (#4351) @dependabot
- ci(deps): bump myrotvorets/info-from-package-json-action from 0.0.2 to 1.1.0 (#4348) @dependabot
- ci(deps): bump path-parse from 1.0.5 to 1.0.7 (#4126) @dependabot
- ci(deps): bump tmpl from 1.0.4 to 1.0.5 (#4337) @dependabot

## [v4.4.3](https://github.com/weseek/growi/compare/v4.4.2...v4.4.3) - 2021-09-17

### 💎 Features

- feat: Slack command permission for each channel (#4302) @hakumizuki

### 🚀 Improvement

- imprv: Copy buttons interaction (#4303) @maow89126
- imprv: Use socket.io room (#4307) @hakumizuki
- imprv: Recent changes of sidebar (#4293) @kaho819

### 🐛 Bug Fixes

- fix: Response to slackbot when no pages are found /growi search (#4321) @stevenfukase
- fix: Keyword highlight styling in search result (#4312) @maow89126
- fix(slackbot-proxy): Call next in connect-styled middleware (#4286) @yuki-takei

### 🧰 Maintenance

- chore: Refactor for "Slack command permission for each channel" (#4295) @zahmis
- support: Update APIv3 docs for pages (#4280) @yuki-takei

## [v4.4.2](https://github.com/weseek/growi/compare/v4.4.0...v4.4.2) - 2021-09-07

### Changes

- Release v4.4.1 (#4262) @github-actions

### 🐛 Bug Fixes

- fix: Plugin backend's permission (#4271) @yuki-takei

### 🧰 Maintenance

- support: Make lerna mode fixed (#4274) @yuki-takei
- support: Make lerna mode fixed (#4263) @yuki-takei

## v4.4.1 (Missing number)

## [v4.4.0](https://github.com/weseek/growi/compare/v4.3.3...v4.4.0) (Discontinued) - 2021-09-06

### Changes

### BREAKING CHANGES

- Official plugins are now preinstalled
- It is no longer compatible with previous versions of official bots

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/44x.html](https://docs.growi.org/en/admin-guide/upgrading/44x.html)

### 💎 Features

- feat: Password resetting by users (#4135) @kaoritokashiki
- feat: Copy bug report btn (#4200) @Mxchaeltrxn
- Feature: User trigger notification and Global notification are available by new Slack integration

### 🚀 Improvement

- imprv: Add attachment button in editor navbar
- imprv: Modified with proxy app installation tutorial (#4174) @hakumizuki
- imprv: Slackbot proxy respone (#4194) @hakumizuki
- imprv: Slackbot proxy respone (#4175, #4201) @zahmis
- Imprv: Admin slack integration (#4190) @yuki-takei

### 🐛 Bug Fixes

- fix: Recursive rename operation from `/parent` to `/parent/child` (#4101) @miya
- fix: adminRequired middleware for socket.io (#4245) @yuki-takei
- fix: Encode spaces in page path in LinkEditModal

### 🧰 Maintenance

- support: Supress warnings for mongo (#4247) @yuki-takei
- support: Add bump-versions script (#4241) @yuki-takei
- support: New release workflow (#4236) @yuki-takei
- Support: Create @growi/core package
- Support: Create @growi/ui package
- Support: Include official plugins as sub packages
- Support: Upgrade libs

## v4.3.3

- Improvement: Welcome page markdown
- Fix: Some recursive operation exclude descendant pages that are restricted for groups (Rename / Delete / Delete completely / Put back / Duplicate)
- Fix: Layout is broken when editing users page ([#4128](https://github.com/weseek/growi/issues/4128))
- Support: Upgrade libs

## v4.3.2

- Feature: Hufflpuff theme
- Improvement: CodeMirror header styles
- Improvement: CodeMirror syntax-highlighting fenced code blocks
- Improvement: Slack Integration Settings: Error behavior when getting connection statuses
- Improvement: Slack Integration Settings: Add links to docs
- Improvement: /_api/v3/recent can be accessed with access token
- Support: Using http-errors

## v4.3.1

- Fix: Build script for production

## v4.3.0

### BREAKING CHANGES

- GROWI manages dependencies with `lerna`: Use `npx lerna bootstrap` instead of `yarn install`
- GROWI includes some official plugins in default: Users no longer need to install [growi-plugin-lsx](https://github.com/weseek/growi-plugin-lsx), [growi-plugin-pukiwiki-like-linker](https://github.com/weseek/growi-plugin-pukiwiki-like-linker) and [growi-plugin-attachment-refs](https://github.com/weseek/growi-plugin-attachment-refs) before build client.

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/43x.html](https://docs.growi.org/en/admin-guide/upgrading/43x.html)

### Updates

- Feature: Enable/Disable option for share link
- Feature: Re-send invitation mail from user management page
- Improvement: Mark users who failed to send invitation emails
- Fix: lsx plugin in the custom sidebar does not work when showing search result page
- Support: Switch the official docker base image from Alpine based to Ubuntu based
- Support: Upgrade libs

## v4.2.21

- Improvement: Headers style on built-in editor
- Improvement: Codemirror is now scrollable one editor height of empty space into view at the bottom of the editor
- Improvement: Upgrade mongodb driver to fix [NODE-2784](https://jira.mongodb.org/browse/NODE-2784)
- Support: Upgrade libs

## v4.2.20

- Improvement: Error message when the password is too short
- Improvement: Repeat XSS processing as a countermeasure against nesting
- Fix: NoSQL injection of access-token-parser
- Fix: Checking permission when operating share links
- Fix: Invalid NaN label is shown when deletedAt of the page is undefined (Introduced by v4.2.8)

## v4.2.19

- Feature: Set max-age of the user's cookie with the env var `SESSION_MAX_AGE`
- Feature: Set max-age of the user's cookie in admin page
- Improvement: Change the first accessing page after installation to the top page
- Support: Upgrade libs

## v4.2.18

- Feature: Convertible page contents width
- Fix: Group selector of User Group Delete Modal does not show all groups
- Fix: Global notification to Slack does not encode spaces of page path
- Support: Upgrade libs

## v4.2.17

- Improvement: Invoke garbage collection when reindex all pages by elasticsearch
- Improvement: Hide Sidebar at shared pages
- Fix: No unsaved alert is displayed without difference the latest markdown and editor value
- Support: Update libs

## v4.2.16

- Fix: "Only inside the group" causes an error (Introduced by v4.2.15)

## v4.2.15

- Improvement: toastr location for editing
- Improvement: Handsontable with static backdrop to prevent from closing when backdrop is clicked
- Fix: Accept invalid page path like `..%2f`
- Fix: Pages updated date is corrupted after recursive operation (Introduced by v4.2.8)
- Support: Upgrade libs

## v4.2.14

- Feature: Add an option to restrict publishing email property for new users
- Improvement: Invite modal in admin page without email server settings
- Improvement: Global notification settings in admin page without email server settings
- Fix: Can create pages on the share route (Introduced by v4.2.8)
- Fix: Pages restrected by group are excluded for recurrence operation (Introduced by v4.2.8)
- Fix: Rename and duplicate to descendants path does not work correctly (Introduced by v4.2.8)
- Support: Update libs

## v4.2.13

- Feature: Detect indent size automatically
- Fix: Some API responses includes email unintentionally
- Fix: An error always displayed in admin pages

## v4.2.12

- Feature: Custom Sidebar
- Fix: Set language correctly for draw.io (diagrams.net)

## v4.2.11

- Fix: Rename decendants is not working (Introduced by v4.2.8)

## v4.2.10

- Feature: Staff Credits for apps on GROWI.cloud
- Improvement: Hackmd button behavior when disabled
- Improvement: Layout of comparing revisions
- Fix: Empty trash is not working

## v4.2.9

- Feature: Comparing revisions
- Improvement: Memory consumption when re-indexing for full text searching
- Improvement: Site URL settings valildation
- Fix: Show comfirmation when transiting page without save
- Fix: Save slack channels history when user trigger notification is invoked
- Fix: The label of alerts for move/rename/delete are borken

## v4.2.8

- Improvement: Performance for pages to rename/duplicate/delete/revert pages
- Fix: Preview scrollbar doesn't sync to editor (Introduced by v4.2.6)
- Fix: Failed to save temporaryUrlCached with using gcs (Introduced by v4.2.3)
- Fix: Fixed not being able to update ses settings (Introduced by v4.2.0)
- Fix: Fixed the display of updtedAt and createdAt being reversed
- Fix: Pass app title value through the XSS filter

## v4.2.7

- Fix: Installer doesn't work on Chrome

## v4.2.6

- Feature: Add a button to jump to Comments section
- Feature: Paste Bootstrap4 Grid HTML with GUI
- Feature: Disable auto formating table option
- Improvement: Layout of Edit Link Modal
- Improvement: Focus to the first input when modal is opened
- Improvement: Preview layout in edit mode
- Improvement: Install process under redundant environment
- Improvement: Add contributors
- Fix: Upgrading to v4.x failed when the user uses Kibela Layout (Introduced by v4.2.0)
- Fix: diagrams.net (draw.io) errors
- Fix: Navbar is not rendered on old iOS
- Support: Expose metrics with Promster
- Support: Upgrade libs

## v4.2.5

- Improvement: Invoke garbage collection when reindex all pages by elasticsearch
- Fix: MathJax rendering does not work

## v4.2.4

- Fix: Fixed an error when creating a new page with `Ctrl-S` (Introduced by v4.2.2)
- Fix: Fixed a strange diff in PageHistory due to Pagination
- Fix: Fixed that the user group page could not be found when using api from the outside

## v4.2.3

- Feature: Insert/edit links with GUI
- Feature: Auto reconnecting to search service
- Improvement: New style of params for Healthcheck API
- Fix: Referencing attachments when `FILE_UPLOAD_DISABLED` is true
- Fix: The message of timeline for restricted pages
- Fix: Parameter validation for Import/Export Archive API
- Fix: Prevent regexp for Search Tags API
- Fix: Add `Content-Security-Policy` when referencing attachments
- Fix: Sanitize at presentation time
- Fix: Remove page path string from message for page lists and timeline when there is no contents

## v4.2.2

- Fix: Consecutive save operations with built-in editor fail (Introduced by v4.2.1)

## v4.2.1

- Fix: Consecutive save operations with HackMD fail (Introduced by v4.2.0)
- Fix: Switching theme to kibela fail (Introduced by v4.2.0)

## v4.2.0

### BREAKING CHANGES

- GROWI v4.2.x no longer support Kibela layout. Kibela theme is newly added and the configuration will migrate to it automatically.

### Updates

- Feature: File Upload Settings on admin pages
- Improvement: Basic layout of page
- Support: Support MongoDB 4.0, 4.2 and 4.4
- Support: Upgrade libs

## v4.1.13

- Fix: MathJax rendering does not work

## v4.1.12

- Fix: Adjust line-height for pre under li
- Fix: Emptying trash process is broken

## v4.1.11

- Improvement: Generating draft DOM id strategy
- Fix: GROWI version downgrade causes a validation error for user.lang

## v4.1.10

- Fix: Make listing users API secure
- Fix: Error message when the server denies guest user connecting with socket.io

## v4.1.9

- Feature: Environment variables to set max connection size to deliver push messages to all clients

## v4.1.8

- Improvement: Rebuilding progress bar colors for Full Text Search Management
- Improvement: Support operations on page data with a null value for author

## v4.1.7

- Improvement: Fire global notification when a new page is created by uploading file
- Fix: Change default `DRAWIO_URI` to embed.diagrams.net
- Fix: An unhandled rejection occures when a user who does not send referer accesses

## v4.1.6

- Improvement: Hide Fab at admin pages
- Fix: Presentation does not work
- Fix: Update GrantSelector status when uploading a file to a new page
- Fix: CopyDropdown origin refs draw.io host wrongly

## v4.1.5

- Feature: Independent S3 configuration and SES configuration for AWS
- Fix: Author name does not displayed in page history
- Fix: Hide unnecessary component when pringing

## v4.1.4 (Discontinued)

## v4.1.3

- Feature: Create/edit linker with GUI
- Improvement: Paging page histories
- Improvement: Avoid using `cursor.snapshot()` in preparation for MongoDB version upgrade
- Improvement: Allow to save "From e-mail address" only in App Settings
- Improvement: Allow to empty "From e-mail address" in App Settings
- Improvement: Export/Import archive data serially so as not to waste memory
- Fix: To be able to delete attachment metadata even when the actual data does not exist
- Fix: Limit the attrubutes of user data for `/_api/v3/users`
- Fix: Prevent XSS with SVG
- Upgrade libs

## v4.1.2

- Fix: Uploaded images do not displayed (Introduced by v4.1.1)

## v4.1.1

- Feature: External share link
- Improvement: Optimize some features that operate revision data
- Fix: Cmd+c/v/... does not work on Mac (Introduced by v4.1.0)
- Fix: "Append params" switch of CopyDropdown does not work when multiple CopyDropdown instance exists
- Fix: "Append params" switch of CopyDropdown escapes spaces
- Fix: Blockdiag does not be rendered
- Fix: Access token parser

## v4.1.0

### BREAKING CHANGES

- GROWI v4.1.x no longer support Node.js v10.x
- GROWI v4.1.x no longer support growi-plugin-attachment-refs@v1

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/41x.html](https://docs.growi.org/en/admin-guide/upgrading/41x.html)

### Updates

- Feature: Server settings synchronization for multiple GROWI Apps
- Feature: Page status alert synchronization for multiple GROWI Apps
- Feature: Smooth scroll for anchor links
- Feature: Mirror Mode with [Konami Code](https://en.wikipedia.org/wiki/Konami_Code)
- Improvement: Determine whether the "In Use" badge is displayed or not by attachment ID
- Improvement: draw.io under NO_CDN environment
- Fix: Deleting/renaming with recursive option affects pages that are inaccessible to active users
- Fix: DrawioModal cuts without beginning/ending line
- Fix: New settings of SMTP and AWS SES are not reflected when server is running
- Fix: Sidebar layout broken when using Kibela layout
- Support: Support Node.js v14
- Support: Update libs

## v4.0.11

- Fix: Fab on search result page does not displayed
- Fix: Adjust margin/padding for search result page
- Fix: PageAlert broken (Introduced by v4.0.9)

## v4.0.10

- Improvement: Adjust ToC height
- Fix: Fail to rename/delete a page set as "Anyone with the link"

## v4.0.9

- Feature: Detailed configurations for OpenID Connect
- Improvement: Navigations
- Improvement: Sticky admin navigation
- Fix: Reseting password doesn't work
- Fix: Styles for printing
- Fix: Unable to create page with original path after emptying trash
- I18n: Support zh-CN

## v4.0.8 (Discontinued)

## v4.0.7

- Feature: Set request timeout for Elasticsearch with env var `ELASTICSEARCH_REQUEST_TIMEOUT`
- Improvement: Apply styles faster on booting client
- Fix: Styles are not applyed on installer
- Fix: Remove last-resort `next()`
- Fix: Enable/disable Notification settings couldn't change when either of the params is undefined
- Fix: Text overflow

## v4.0.6

- Fix: Avatar images in Recent Changes are not shown
- Fix: Full screen modal of Handsontable and Draw.io don't work
- Fix: Shortcut for creating page respond with modifier key wrongly (Introduced by v4.0.5)

## v4.0.5

- Improvement: Return pre-defined session id when healthcheck
- Improvement: Refactor caching for profile image
- Improvement: Layout for global search help on mobile
- Improvement: Layout for confidential notation
- Fix: Shortcut for creating page doesn't work
- Support: Dev in container
- Support: Upgrade libs

## v4.0.4

- Feature: Drawer/Dock mode selector
- Improvement: Admin pages navigation
- Improvement: Ensure not to avoid session management even when accessing to healthcheck
- Support: Refactor unstated utils
- Support: Upgrade libs

## v4.0.3

- Feature: Copy page path dropdown with Append params switch
- Improvement: Truncate overflowed user browsing history
- Improvement: Tabs appearance on mobile
- Improvement: Search help appearance on mobile
- Improvement: Accessibility of login page
- Fix: Editor was broken by long lines
- Fix: Editor doesn't work on mobile
- Fix: Word break in Recent Updated contents
- Fix: navbar is broken on Safari

## v4.0.2

- Fix: Internal Server Error occurred when the guest user access to the pages that has likes
- Fix: Some buttons are broken on Safari

## v4.0.1

- Improvement: Accessibility for Handsontable under dark mode
- Improvement: Refactor '/pages.exist' API
- Fix: Storing the state of sidebar
- Fix: Comments order should be asc
- Fix: Show/Hide replies button doesn't work
- Fix: Tooltip doesn't work
- Fix: Change the display of the scroll bar when modal is shown
- Fix: Submit with enter key on Create/Rename modals
- Fix: Show/Hide Unlink redirection button conditions
- Fix: Link color in alerts
- Support: Upgrade libs

## v4.0.0

### BREAKING CHANGES

- Crowi Classic Behavior is removed
- Crowi Classic Layout is removed
- 'default-dark' theme is now merged as a dark mode variant of 'default' theme
- 'blue-night' theme is now merged as a dark mode variant of 'mono-blue' theme

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/40x.html](https://docs.growi.org/en/admin-guide/upgrading/40x.html)

### Updates

- Feature: Sidebar
- Feature: Recent changes on Sidebar
- Feature: Switch Light/Dark Mode
- Improvement: Migrate to Bootstrap 4
- Improvement: Copy Page URL menu item to copy path dropdown
- Improvement: Show contributors by Bootstrap Modal
- Support: Upgrade libs

## v3.8.1

### BREAKING CHANGES

- Now Elasticsearch requires the privilege `cluster:monitor/health` instead of `cluster:monitor/nodes/info`

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/38x.html](https://docs.growi.org/en/admin-guide/upgrading/38x.html)

### Updates

- Improvement: Change the health check method for Elasticsearch
- Fix: Unset overflow-y style for Edit Tags Modal
- Fix: Duplicate page source is overwrited (Introduced by 3.7.6)

## v3.8.0  (Discontinued)

## v3.7.7

- Feature: Empty trash pages
- Improvement: Behavior of Reconnect to Elasticsearch button
- Fix: Duplicate page source is overwrited (Introduced by 3.7.6)

## v3.7.6  (Discontinued)

## v3.7.5

- Fix: Draw.io diagrams rendered twice
- Fix: Behavior of password reset modal is strange
- Fix: Import GROWI Archive doesn't restore some data correctly
- Fix: Attachments list on root page and users top pages
- Fix: Trash page is no longer editable
- Fix: Rendering Timeline on /trash

## v3.7.4

- Fix: Broken by displaying user image

## v3.7.3

- Feature: Profile Image Cropping
- Improvement: Reactify users pages
- Improvement: Detect language and adjust the order of first and last names when creating accounts in OAuth
- Fix: Installation is broken when selecting Japanese (Introduced by 3.7.0)
- Fix: Mathjax Rendering is unstable (workaround) (Introduced by 3.7.0)
- Fix: Notification Setting couldn't update without slack token (Introduced by 3.6.6)
- Support: Add GROWI Contributers

## v3.7.2

- Feature: User Management Filtering/Sort
- Feature: Show env vars on Admin pages
- Fix: Attachment row z-index
- I18n: HackMD integration alert

## v3.7.1

- Improvement: Add an option that make it possible to choose what to send notifications
- Improvement: Add the env var `DRAWIO_URI`
- Improvement: Accessibility for 'spring' theme
- Improvement: Editor scroll sync behaves strangely when using draw.io blocks
- Fix: Coudn't upload file on Comment Editor (Introduced by 3.5.8)
- I18n: HackMD integration

## v3.7.0

### BREAKING CHANGES

None.

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/37x.html](https://docs.growi.org/en/admin-guide/upgrading/37x.html)

### Updates

- Feature: [Draw.io](https://www.draw.io/) Integration
- Feature: SAML Attribute-based Login Control
- Improvement: Reactify admin pages (Security)
- Improvement: Behavior of pre-editing screen of HackMD when user needs to resume

## v3.6.10

- Fix: Redirect logic for users except for actives (Introduced by 3.6.9)

## v3.6.9

- Improvement: Redirection when login/logout
- Improvement: Add home icon before '/'
- Fix: Client crashed when the first login (Introduced by 3.6.8)

## v3.6.8

- Improvement: Show page history side-by-side
- Improvement: Optimize markdown rendering
- Improvement: Reactify admin pages (Navigation)
- Fix: Reply comments collapsed are broken (Introduced by 3.6.7)
- Support: Update libs

## v3.6.7

- Feature: Anchor link for comments
- Improvement: Show error toastr when saving page is failed because of empty document
- Fix: Admin Customise couldn't restore stored config value (Introduced by 3.6.2)
- Fix: Admin Customise missed preview functions (Introduced by 3.6.2)
- Fix: AWS doesn't work (Introduced by 3.6.4)
- Fix: Ensure not to get unrelated indices information in Elasticsearch Management (Introduced by 3.6.6)
- Support: Optimize bundles
- Support: Optimize build-prod job with caching node_modules/.cache

## v3.6.6

- Feature: Reconnect to Elasticsearch from Full Text Search Management
- Feature: Normalize indices of Elasticsearch from Full Text Search Management
- Improvement: Add 'spring' theme
- Improvement: Reactify admin pages (Notification)
- Impromvement: Add `checkMiddlewaresStrictly` option to Healthcheck API
- Improvement: Accessibility for History component under dark themes
- Fix: Warning on client console when developing /admin/app
- Support: Upgrade libs

## v3.6.5 (Discontinued)

## v3.6.4

- Feature: Alert for stale page
- Improvement: Reactify admin pages (Home)
- Improvement: Reactify admin pages (App)
- Improvement: Accessibility for editor icons of dark themes
- Improvement: Accessibility for importing table data pane
- Improvement: Resolve username and email when logging in with Google OAuth

## v3.6.3

- Improvement: Searching users in UserGroup Management
- Fix: Repair google authentication by migrating to jaredhanson/passport-google-oauth2
- Fix: Markdown Settings are broken by the button to import recommended settings
- Support: Upgrade libs

## v3.6.2

- Improvement: Reactify admin pages (Customize)
- Improvement: Ensure not to consider `[text|site](https://example.com]` as a row in the table
- Improvement: Enter key behavior in markdown table
- Fix: Pre-installed plugins in official docker image are not detected (Introduced by 3.6.0)
- Fix: Emoji Autocomplete window does not float correctly (Introduced by 3.5.0)

## v3.6.1

### BREAKING CHANGES

- GROWI v3.6.x no longer support Node.js v8.x
- The name of database that is storing migrations meta data has been changed. This affects **only when `MONGO_URI` has parameters**. v3.5.x or above has a bug ([#1361](https://github.com/weseek/growi/issues/1361))

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/36x.html](https://docs.growi.org/en/admin-guide/upgrading/36x.html)

### Updates

- Improvement: Drop unnecessary MongoDB collection indexes
- Improvement: Accessibility of Antarctic theme
- Improvement: Reactify admin pages (Markdown Settings)
- Fix: Appending tag is failed by wrong index of PageTagRelation (Introduced by 3.5.20)
- Fix: Pages without heading slash is invalid but creatable
- Fix: Connect to Elasticsearch with `httpAuth` param
- Support: Support Node.js v12
- Support: Optimize build in dev with hard-source-webpack-plugin
- Support: Upgrade libs

## v3.6.0 (Discontinued)

## v3.5.25

- Improvement: Disable ESC key to close Handsontable Modal
- Fix: Exported data of empty collection is broken
- Fix: Some components crash after when the page with attachment has exported/imported

## v3.5.24

- Fix: Plugins are not working on Heroku

## v3.5.23

- Fix: Global Notification failed to send e-mail
- Fix: Pagination is not working for trash list
- Fix: Healthcheck API with `?connectToMiddlewares` returns error
- Support: Upgrade libs

## v3.5.22

- Improvement: Add `FILE_UPLOAD_DISABLED` env var

## v3.5.21

- Improvement: Cache control when retrieving attachment data
- Fix: Inviting user doesn't work (Introduced by 3.5.20)

## v3.5.20

- Improvement: Organize MongoDB collection indexes uniqueness
- Improvement: Reactify admin pages (External Account Management)
- Fix: Search result or Timeline shows loading icon eternally when retrieving not accessible page
- Support: Use SearchBox Elasticsearch Addon on Heroku
- Support: Upgrade libs

## v3.5.19 (Discontinued)

## v3.5.18

- Improvement: Import GROWI Archive
- Improvement: Optimize handling promise of stream when exporting archive
- Improvement: Optimize handling promise of stream when building indices
- Improvement: Add link to [docs.growi.org](https://docs.growi.org)
- Fix: Monospace font code is broken when printing on Mac

## v3.5.17

- Feature: Upload to GCS (Google Cloud Storage)
- Feature: Statistics API
- Improvement: Optimize exporting
- Improvement: Show progress bar when exporting
- Improvement: Validate collection combinations when importing
- Improvement: Reactify admin pages
- Fix: Use HTTP PlantUML URL in default (Introduced by 3.5.12)
- Fix: Config default values
- Support: REPL with `console` npm scripts

## v3.5.16

- Fix: Full Text Search doesn't work after when building indices (Introduced by 3.5.12)

## v3.5.15

- Feature: Import/Export Page data
- Fix: The link to Sandbox on Markdown Help Modal doesn't work
- Support: Upgrade libs

## v3.5.14 (Discontinued)

## v3.5.13

- Feature: Re-edit comments
- Support: [growi-plugin-attachment-refs](https://github.com/weseek/growi-plugin-attachment-refs)
- Support: Upgrade libs

## v3.5.12

- Improvement: Use Elasticsearch Alias
- Improvement: Connect to HTTPS PlantUML URL in default
- Fix: Global Notification doesn't work after updating Webhook URL
- Fix: User Trigger Notification is not be sent when channel is not specified
- Support: Upgrade libs

## v3.5.11

- Fix: HackMD Editor shows 404 error when HackMD redirect to fqdn URI (Introduced by 3.5.8)
- Fix: Timeline doesn't work (Introduced by 3.5.1)
- Fix: Last Login field does not shown in /admin/user
- Support: Upgrade libs

## v3.5.10

- Feature: Send Global Notification with Slack
- Improvement: Show loading spinner when fetching page history data
- Improvement: Hierarchical page link when the page is in /Trash
- Fix: Code Highlight Theme does not change (Introduced by 3.5.2)
- Support: Upgrade libs

## v3.5.9

- Fix: Editing table with Spreadsheet like GUI (Handsontable) is failed
- Fix: Plugins are not initialized when first launching (Introduced by 3.5.0)
- Support: Upgrade libs

## v3.5.8

- Improvement: Controls when HackMD/CodiMD has unsaved draft
- Improvement: Show hints if HackMD/CodiMD integration is not working
- Improvement: GROWI server obtains HackMD/CodiMD page id from the 302 response header
- Improvement: Comment Thread Layout
- Improvement: Show commented date with date distance format

## v3.5.7 (Discontinued)

## v3.5.6

- Fix: Saving new page is failed when empty string tag is set
- Fix: Link of Create template page button in New Page Modal is broken
- Fix: Global Notification dows not work when creating/moving/deleting/like/comment

## v3.5.5

- Feature: Support S3-compatible object storage (e.g. MinIO)
- Feature: Enable/Disable ID/Password Authentication
- Improvement: Login Mechanism with HTTP Basic Authentication header
- Improvement: Reactify Table Of Contents
- Fix: Profile images are broken in User Management
- Fix: Template page under root page doesn't work
- Support: Upgrade libs

## v3.5.4

- Fix: List private pages wrongly
- Fix: Global Notification Trigger Path does not parse glob correctly
- Fix: Consecutive page deletion requests cause unexpected complete page deletion

## v3.5.3

- Improvement: Calculate string width when save with Spreadsheet like GUI (Handsontable)
- Fix: Search Result Page doesn't work
- Fix: Create/Update page API returns data includes author's password hash
- Fix: Dropdown to copy page path/URL/MarkdownLink shows under CodeMirror vscrollbar
- Fix: Link to /trash in Dropdown menu

## v3.5.2

- Feature: Remain metadata option when Move/Rename page
- Improvement: Support code highlight for Swift and Kotlin
- Fix: Couldn't restrict page with user group permission
- Fix: Couldn't duplicate a page when it restricted by a user group permission
- Fix: Consider timezone on admin page
- Fix: Editor doesn't work on Microsoft Edge
- Support: Upgrade libs

## v3.5.1

### BREAKING CHANGES

- GROWI no longer supports Protection system with Basic Authentication
- GROWI no longer supports Crowi Classic Authentication Mechanism
- GROWI no lonnger supports plugins with schema version 2
- The restriction mode of the root page (`/`) will be set 'Public'
- The restriction mode of the root page (`/`) can not be changed after v 3.5.1

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/35x.html](https://docs.growi.org/en/admin-guide/upgrading/35x.html)

### Updates

- Feature: Comment Thread
- Feature: OpenID Connect authentication
- Feature: HTTP Basic authentication
- Feature: Staff Credits with [Konami Code](https://en.wikipedia.org/wiki/Konami_Code)
- Feature: Restricte Complete Deletion of Pages
- Improvement Draft list
- Fix: Deleting page completely
- Fix: Search with `prefix:` param with CJK pathname
- Fix: Could not edit UserGroup even if `PUBLIC_WIKI_ONLY` is not set
- I18n: User Management Details
- I18n: Group Management Details
- Support: Apply unstated
- Support: Use Babel 7
- Support: Support plugins with schema version 3
- Support: Abolish Old Config API
- Support: Apply Jest for Tests
- Support: Upgrade libs

## v3.5.0 (Discontinued)

## v3.4.7

- Improvement: Handle private pages on group deletion
- Fix: Searching with `tag:xxx` syntax doesn't work
- Fix: Check CSRF when updating user data
- Fix: `createdAt` field initialization
- I18n: Import data page
- I18n: Group Management page

## v3.4.6

- Feature: Tags
- Feature: Dropdown to copy page path/URL/MarkdownLink
- Feature: List of drafts
- Improvement: Replace icons of Editor Tool Bar
- Improvement: Show display name when mouse hover to user image
- Fix: URL in slack message is broken on Safari
- Fix: Registration does not work when basic auth is enabled
- Support: Publish API docs with swagger-jsdoc and ReDoc
- Support: Upgrade libs

## v3.4.5

- Improvement: Pass autolink through the XSS filter according to CommonMark Spec
- Fix: Update ElasticSearch index when deleting/duplicating pages
- Fix: Xss filter breaks PlantUML arrows
- Support: Support growi-plugin-lsx@2.2.0
- Support: Upgrade libs

## v3.4.4

- Fix: Comment component doesn't work

## v3.4.3

- Improvement: Add 'antarctic' theme
- Support Apply eslint-config-airbnb based rules
- Support Apply prettier and stylelint
- Support: Upgrade libs

## v3.4.2

- Fix: Nofitication to Slack doesn't work (Introduced by 3.4.0)

## v3.4.1

- Fix: "Cannot find module 'stream-to-promise'" occured when build client with `FILE_UPLOAD=local`

## v3.4.0

### BREAKING CHANGES

None.

Upgrading Guide: [https://docs.growi.org/en/admin-guide/upgrading/34x.html](https://docs.growi.org/en/admin-guide/upgrading/34x.html)

### Updates

- Improvement: Restrict to access attachments when the user is not allowed to see page
- Improvement: Show fans and visitors of page
- Improvement: Full text search tokenizing
- Improvement: Markdown comment on Crowi Classic Layout
- Fix: Profile image is not displayed when `FILE_UPLOAD=mongodb`
- Fix: Posting comment doesn't work under Crowi Classic Layout (Introduced by 3.1.5)
- Fix: HackMD doesn't work when `siteUrl` ends with slash
- Fix: Ensure not to be able to move/duplicate page to the path which has trailing slash
- Support: Launch with Node.js v10
- Support: Launch with MongoDB 3.6
- Support: Launch with Elasticsearch 6.6
- Support: Upgrade libs

## v3.3.10

- Feature: PlantUML and Blockdiag on presentation
- Improvement: Render slides of presentation with GrowiRenderer
- Fix: Unportalizing doesn't work
- Support: Use mini-css-extract-plugin instead of extract extract-text-webpack-plugin
- Support: Use terser-webpack-plugin instead of uglifyjs-webpack-plugin
- Support: Upgrade libs

## v3.3.9

- Fix: Import from Qiita:Team doesn't work (Introduced by 3.3.0)
- Fix: Typeahead shows autocomplete wrongly (Introduced by 3.3.8)
- Support: Upgrade libs

## v3.3.8

- Fix: Move/Duplicate don't work (Introduced by 3.3.7)
- Fix: Server doesn't respond when root page is restricted
- Support: Upgrade libs

## v3.3.7

- Feature: Editor toolbar
- Feature: `prefix:/path` searching syntax to filter with page path prefix
- Feature: Add an option to filter only children to searching box of navbar
- Improvement: Suggest page path when moving/duplicating/searching
- Fix: Anonymous users couldn't search (Introduced by 3.3.6)
- I18n: Searching help
- Support: Prepare to suppoert Node.js v10
- Support: Upgrade libs

## v3.3.6

- Improvement: Site URL settings must be set
- Improvement: Site URL settings can be set with environment variable
- Fix: "Anyone with the link" ACL doesn't work correctly (Introduced by 3.3.0)
- Fix: Related pages list of /admin/user-group-detail/xxx doesn't show anything (Introduced by 3.3.0)
- Fix: Diff of revision contents doesn't appeared when notifing with slack
- Fix: NPE occured on /admin/security when Crowi Classic Auth Mechanism is set
- Fix: Coudn't render Timing Diagram with PlantUML
- I18n: Cheatsheet for editor
- I18n: Some admin pages
- Support: Upgrade libs

## v3.3.5 (Discontinued)

## v3.3.4

- Improvement: SAML configuration with environment variables
- Improvement: Upload file with pasting from clipboard
- Fix: `/_api/revisions.get` doesn't populate author data correctly
- Fix: Wrong OAuth callback url are shown at admin page
- Fix: Connecting to MongoDB failed when processing migration
- Support: Get ready to use new config management system

## v3.3.3

- Feature: Show line numbers to a code block
- Feature: Bulk update the scope of descendant pages when create/update page
- Improvement: The scope of ascendant page will be retrieved and set to controls in advance when creating a new page
- Fix: Pages that is restricted by groups couldn't be shown in search result page
- Fix: Pages order in search result page was wrong
- Fix: Guest user can't search
- Fix: Possibility that ExternalAccount deletion processing selects incorrect data
- Support: Upgrade libs

## v3.3.2

- Fix: Specified Group ACL is not persisted correctly (Introduced by 3.3.0)

## v3.3.1

- Feature: NO_CDN Mode
- Feature: Add option to show/hide restricted pages in list
- Feature: MongoDB GridFS quota
- Improvement: Refactor Access Control
- Improvement: Checkbox behavior of task list
- Improvement: Fixed search input on search result page
- Improvement: Add 'christmas' theme
- Improvement: Select default language of new users
- Fix: Hide restricted pages contents in timeline
- Support: Upgrade libs

## v3.3.0 (Discontinued)

## v3.2.10

- Fix: Pages in trash are available to create
- Fix: Couldn't create portal page under Crowi Classic Behavior
- Fix: Table tag in Timeline/SearchResult missed border and BS3 styles
- I18n: Installer

## v3.2.9

- Feature: Attachment Storing to MongoDB GridFS
- Fix: row/col moving of Spreadsheet like GUI (Handsontable) doesn't work
- Fix: Emoji AutoComplete dialog pops up at wrong position
- Support: Upgrade libs

## v3.2.8

- Improvement: Add an option to use email for account link when using SAML federation
- Fix: Editor layout is sometimes broken
- Fix: Normalize table data for Spreadsheet like GUI (Handsontable) when import
- Support: Improve development environment
- Support: Upgrade libs

## v3.2.7

- Feature: Import CSV/TSV/HTML table on Spreadsheet like GUI (Handsontable)
- Fix: Pasting table data copied from Excel includes unnecessary line breaks
- Fix: Page break Preset 1 for Presentation mode is broken
- Fix: Login Form when LDAP login failed caused 500 Internal Server Error

## v3.2.6

- Feature: Add select alignment buttons of Spreadsheet like GUI (Handsontable)
- Improvement: Shrink the rows that have no diff of revision history page
- Fix: Login form rejects weak password
- Fix: An error occured by uploading attachment file when the page is not exists (Introduced by 2.3.5)
- Support: Upgrade libs

## v3.2.5

- Improvement: Expandable Spreadsheet like GUI (Handsontable)
- Improvement: Move/Resize rows/columns of Spreadsheet like GUI (Handsontable)
- Improvement: Prevent XSS of New Page modal
- Fix: Recent Created tab of user home shows wrong page list (Introduced by 3.2.4)
- Support: Upgrade libs

## v3.2.4

- Feature: Edit table with Spreadsheet like GUI (Handsontable)
- Feature: Paging recent created in users home
- Improvement: Specify certificate for SAML Authentication
- Fix: SAML Authentication didn't work (Introduced by 3.2.2)
- Fix: Failed to create new page with title which includes RegEx special characters
- Fix: Preventing XSS Settings are not applied in default (Introduced by 3.1.12)
- Support: Mongoose migration mechanism
- Support: Upgrade libs

## v3.2.3

- Feature: Kibela like layout
- Improvement: Custom newpage separator for presentation view
- Support: Shrink image size for themes which recently added

## v3.2.2

- Feature: SAML Authentication (SSO)
- Improvement: Add 'wood' theme
- Improvement: Add 'halloween' theme
- Improvement: Add 'island' theme
- Fix: Sending email function doesn't work
- Support Upgrade libs

## v3.2.1

- Feature: Import data from esa.io
- Feature: Import data from Qiita:Team
- Feature: Add the endpoint for health check
- Improvement: Adjust styles when printing
- Fix: Renaming page doesn't work if the page was saved with shortcut
- Support: Refactor directory structure
- Support Upgrade libs

## v3.2.0

- Feature: HackMD integration so that user will be able to simultaneously edit with multiple people
- Feature: Login with Twitter Account (OAuth)
- Fix: The Initial scroll position is wrong when reloading the page

## v3.1.14

- Improvement: Show help for header search box
- Improvement: Add Markdown Cheatsheet to Editor component
- Fix: Couldn't delete page completely from search result page
- Fix: Tabs of trash page are broken

## v3.1.13

- Feature: Global Notification
- Feature: Send Global Notification with E-mail
- Improvement: Add attribute mappings for email to LDAP settings
- Support: Upgrade libs

## v3.1.12

- Feature: Add XSS Settings
- Feature: Notify to Slack when comment
- Improvement: Prevent XSS in various situations
- Improvement: Show forbidden message when the user accesses to ungranted page
- Improvement: Add overlay styles for pasting file to comment form
- Fix: Omit unnecessary css link (Introduced by 3.1.10)
- Fix: Invitation mail do not be sent
- Fix: Edit template button on New Page modal doesn't work

## v3.1.11

- Fix: OAuth doesn't work in production because callback URL field cannot be specified (Introduced by 3.1.9)

## v3.1.10

- Fix: Enter key on react-bootstrap-typeahead doesn't submit (Introduced by 3.1.9)
- Fix: CodeMirror of `/admin/customize` is broken (Introduced by 3.1.9)

## v3.1.9

- Feature: Login with Google Account (OAuth)
- Feature: Login with GitHub Account (OAuth)
- Feature: Attach files in Comment
- Improvement: Write comment with CodeMirror Editor
- Improvement: Post comment with `Ctrl-Enter`
- Improvement: Place the commented page at the beginning of the list
- Improvement: Resolve errors on IE11 (Experimental)
- Support: Migrate to webpack 4
- Support: Upgrade libs

## v3.1.8 (Discontinued)

## v3.1.7

- Fix: Update hidden input 'pageForm[grant]' when save with `Ctrl-S`
- Fix: Show alert message when conflict
- Fix: `BLOCKDIAG_URI` environment variable doesn't work
- Fix: Paste in markdown list doesn't work correctly
- Support: Ensure to inject logger configuration from environment variables
- Support: Upgrade libs

## v3.1.6

- Feature: Support [blockdiag](http://blockdiag.com)
- Feature: Add `BLOCKDIAG_URI` environment variable
- Fix: Select modal for group is not shown
- Support: Upgrade libs

## v3.1.5

- Feature: Write comment with Markdown
- Improvement: Support some placeholders for template page
- Improvement: Omit unnecessary response header
- Improvement: Support LDAP attribute mappings for user's full name
- Improvement: Enable to scroll revision-toc
- Fix: Posting to Slack doesn't work (Introduced by 3.1.0)
- Fix: page.rename api doesn't work
- Fix: HTML escaped characters in markdown are unescaped unexpectedly after page is saved
- Fix: sanitize `#raw-text-original` content with 'entities'
- Fix: Double newline character posted (Introduced by 3.1.4)
- Fix: List and Comment components do not displayed (Introduced by 3.1.4)
- Support: Upgrade libs

## v3.1.4 (Discontinued)

## v3.1.3 (Discontinued)

## v3.1.2

- Feature: Template page
- Improvement: Add 'future' theme
- Improvement: Modify syntax for Crowi compatible template feature
- Improvement: Escape iframe tag in block codes
- Support: Upgrade libs

## v3.1.1

- Improvement: Add 'blue-night' theme
- Improvement: List up pages which restricted for Group ACL
- Fix: PageGroupRelation didn't remove when page is removed completely

## v3.1.0

- Improvement: Group Access Control List - Select group modal
- Improvement: Better input on mobile
- Improvement: Detach code blocks correctly
- Improvement: Auto-format markdown table which includes multibyte text
- Improvement: Show icon when auto-format markdown table is activated
- Improvement: Enable to switch show/hide border for highlight.js
- Improvement: BindDN field allows also ActiveDirectory styles
- Improvement: Show LDAP logs when testing login
- Fix: Comment body doesn't break long terms
- Fix: lsx plugin lists up pages that hit by forward match wrongly (Introduced by 3.0.4)
- Fix: Editor is broken on IE11
- Support: Multilingualize React components with i18next
- Support: Organize dependencies
- Support: Upgrade libs

## v3.0.13

- Improvement: Add Vim/Emacs/Sublime-Text icons for keybindings menu
- Improvement: Add 'mono-blue' theme
- Fix: Unportalize process failed silently
- Fix: Sidebar breaks editor layouts
- Support: Switch the logger from 'pino' to 'bunyan'
- Support: Set the alias for 'debug' to the debug function of 'bunyan'
- Support: Translate `/admin/security`
- Support: Optimize bundles
- Support: Optimize .eslintrc.js

## v3.0.12

- Feature: Support Vim/Emacs/Sublime-Text keybindings
- Improvement: Add some CodeMirror themes (Eclipse, Dracula)
- Improvement: Dynamic loading for CodeMirror theme files from CDN
- Improvement: Prevent XSS when move/redirect/duplicate

## v3.0.11

- Fix: login.html is broken in iOS
- Fix: Removing attachment is crashed
- Fix: File-attaching error after new page creation
- Support: Optimize development build
- Support: Upgrade libs

## v3.0.10

- Improvement: Add 'nature' theme
- Fix: Page list and Timeline layout for layout-growi
- Fix: Adjust theme colors (Introduced by 3.0.9)

## v3.0.9

- Fix: Registering new LDAP User is failed (Introduced by 3.0.6)
- Support: Organize scss for overriding bootstrap variables
- Support: Upgrade libs

## v3.0.8

- Improvement: h1#revision-path occupies most of the screen when the page path is long
- Improvement: Ensure not to save concealed email field to localStorage
- Fix: Cannot input "c" and "e" on iOS

## v3.0.7

- Improvement: Enable to download an attached file with original name
- Improvement: Use MongoDB for session store instead of Redis
- Improvement: Update dropzone overlay icons and styles
- Fix: Dropzone overlay elements doesn't show (Introduced by 3.0.0)
- Fix: Broken page path of timeline (Introduced by 3.0.4)

## v3.0.6

- Improvement: Automatically bind external accounts newly logged in to local accounts when username match
- Improvement: Simplify configuration for Slack Web API
- Support: Use 'slack-node' instead of '@slack/client'
- Support: Upgrade libs

## v3.0.5

- Improvement: Update lsx icons and styles
- Fix: lsx plugins doesn't show page names

## v3.0.4

- Improvement: The option that switch whether add h1 section when create new page
- Improvement: Encode page path that includes special character
- Fix: Page-saving error after new page creation

## v3.0.3

- Fix: Login page is broken in iOS
- Fix: Hide presentation tab if portal page
- Fix: A few checkboxes doesn't work
- Fix: Activating invited user form url is wrong
- Support: Use postcss-loader and autoprefixer

## v3.0.2

- Feature: Group Access Control List
- Feature: Add site theme selector
- Feature: Add a control to switch whether email shown or hidden by user
- Feature: Custom title tag content
- Fix: bosai version
- Support: Rename to GROWI
- Support: Add dark theme
- Support: Refreshing bootstrap theme and icons
- Support: Use Browsersync instead of easy-livereload
- Support: Upgrade libs

## v3.0.1 (Discontinued)

## v3.0.0 (Discontinued)

## v2.4.4

- Feature: Autoformat Markdown Table
- Feature: highlight.js Theme Selector
- Fix: The bug of updating numbering list by codemirror
- Fix: Template LangProcessor doesn't work (Introduced by 2.4.0)
- Support: Apply ESLint
- Support: Upgrade libs

## v2.4.3

- Improvement: i18n in `/admin`
- Improvement: Add `SESSION_NAME` environment variable
- Fix: All Elements are cleared when the Check All button in DeletionMode
- Support: Upgrade libs

## v2.4.2

- Improvement: Ensure to set absolute url from root when attaching files when `FILE_UPLOAD=local`
- Fix: Inline code blocks that includes doller sign are broken
- Fix: Comment count is not updated when a comment of the page is deleted
- Improvement: i18n in `/admin` (WIP)
- Support: Upgrade libs

## v2.4.1

- Feature: Custom Header HTML
- Improvement: Add highlight.js languages
- Fix: Couldn't connect to PLANTUML_URI (Introduced by 2.4.0)
- Fix: Couldn't render UML which includes CJK (Introduced by 2.4.0)
- Support: Upgrade libs

## v2.4.0

- Feature: Support Footnotes
- Feature: Support Task lists
- Feature: Support Table with CSV
- Feature: Enable to render UML diagrams with public plantuml.com server
- Feature: Enable to switch whether rendering MathJax in realtime or not
- Improvement: Replace markdown parser with markdown-it
- Improvement: Generate anchor of headers with header strings
- Improvement: Enhanced Scroll Sync on Markdown Editor/Preview
- Improvement: Update `#revision-body` tab contents after saving with `Ctrl-S`
- Fix: 500 Internal Server Error occures when basic-auth configuration is set

## v2.3.9

- Fix: `Ctrl-/` doesn't work on Chrome
- Fix: Close Shortcuts help with `Ctrl-/`, ESC key
- Fix: Jump to last line wrongly when `.revision-head-edit-button` clicked
- Support: Upgrade libs

## v2.3.8

- Feature: Suggest page path when creating pages
- Improvement: Prevent keyboard shortcuts when modal is opened
- Improvement: PageHistory UI
- Improvement: Ensure to scroll when edit button of section clicked
- Improvement: Enabled to toggle the style for active line
- Support: Upgrade libs

## v2.3.7

- Fix: Open popups when `Ctrl+C` pressed (Introduced by 2.3.5)

## v2.3.6

- Feature: Theme Selector for Editor
- Improvement: Remove unportalize button from crowi-plus layout
- Fix: CSS for admin pages
- Support: Shrink the size of libraries to include

## v2.3.5

- Feature: Enhanced Editor by CodeMirror
- Feature: Emoji AutoComplete
- Feature: Add keyboard shortcuts
- Improvement: Attaching file with Dropzone.js
- Improvement: Show shortcuts help with `Ctrl-/`
- Fix: DOMs that has `.alert-info` class don't be displayed
- Support: Switch and upgrade libs

## v2.3.4 (Discontinued)

## v2.3.3

- Fix: The XSS Library escapes inline code blocks
- Fix: NPE occurs on Elasticsearch when initial access
- Fix: Couldn't invite users(failed to create)

## v2.3.2

- Improvement: Add LDAP group search options

## v2.3.1

- Fix: Blockquote doesn't work
- Fix: Couldn't create user with first LDAP logging in

## v2.3.0

- Feature: LDAP Authentication
- Improvement: Prevent XSS
- Fix: node versions couldn't be shown
- Support: Upgrade libs

## v2.2.4

- Fix: googleapis v23.0.0 lost the function `oauth2Client.setCredentials`
- Fix: HeaderSearchBox didn't append 'q=' param when searching

## v2.2.3

- Fix: The server responds anything when using passport
- Fix: Update `lastLoginAt` when login is success
- Support: Replace moment with date-fns
- Support: Upgrade react-bootstrap-typeahead
- Improvement: Replace emojify.js with emojione

## v2.2.2 (Discontinued)

## v2.2.1

- Feature: Duplicate page
- Improve: Ensure that admin users can remove users waiting for approval
- Fix: Modal doesn't work with React v16
- Support: Upgrade React to 16
- Support: Upgrade outdated libs

## v2.2.0

- Support: Merge official Crowi v1.6.3

## v2.1.2

- Improvement: Ensure to prevent suspending own account
- Fix: Ensure to be able to use `.` for username when invited
- Fix: monospace font for `code` tag

## v2.1.1

- Fix: The problem that React Modal doesn't work
- Support: Lock some packages(react, react-dom, mongoose)

## v2.1.0

- Feature: Adopt Passport the authentication middleware
- Feature: Selective batch deletion in search result page
- Improvement: Ensure to be able to login with both of username or email
- Fix: The problem that couldn't update user data in /me
- Support: Upgrade outdated libs

## v2.0.9

- Fix: Server is down when a guest user accesses to someone's private pages
- Support: Merge official Crowi (master branch)
- Support: Upgrade outdated libs

## v2.0.8

- Fix: The problem that path including round bracket makes something bad
- Fix: Recursively option processes also unexpedted pages
- Fix: en_US translation

## v2.0.7

- Improvement: Add recursively option for Delete/Move/Putback operation
- Improvement: Comment layout and sort order (crowi-plus Enhanced Layout)

## v2.0.6

- Fix: check whether `$APP_DIR/public/uploads` exists before creating symlink

## v2.0.5

- Improvement: Adjust styles for CodeMirror
- Fix: File upload does not work when using crowi-plus-docker-compose and `FILE_UPLOAD=local` is set

## v2.0.2 - 2.0.4 (Discontinued)

## v2.0.1

- Feature: Custom Script
- Improvement: Adjust layout and styles for admin pages
- Improvement: Record and show last updated date in user list page
- Fix: Ignore Ctrl+(Shift+)Tab when editing (cherry-pick from the official)

## v2.0.0

- Feature: Enabled to integrate with Slack using Incoming Webhooks
- Support: Upgrade all outdated libs

## v1.2.16

- Improvement: Condition for creating portal
- Fix: Couldn't create new page after installation cleanly

## v1.2.15

- Improvement: Optimize cache settings for express server
- Improvement: Add a logo link to the affix header
- Fix: Child pages under `/trash` are not shown when applying crowi-plus Simplified Behavior

## v1.2.14

- Fix: Tabs(`a[data-toggle=tab][href=#...]`) push browser history twice
- Fix: `a[href=#edit-form]` still save history even when disabling pushing states option

## v1.2.13

- Improvement: Enabled to switch whether to push states with History API when tabs changes
- Fix: Layout of the Not Found page

## v1.2.12 (Discontinued)

## v1.2.11

- Improvement: Enabled to open editing form from affix header
- Improvement: Enabled to open editing form from each section headers

## v1.2.10

- Fix: Revise `server:prod:container` script for backward compatibility

## v1.2.9

- Improvement: Enabled to save with <kbd>⌘+S</kbd> on Mac
- Improvement: Adopt the fastest logger 'pino'
- Fix: The problem that can't upload profile image

## v1.2.8

- Fix: The problem that redirect doesn't work when using 'crowi-plus Simplified Behavior'

## v1.2.7 (Discontinued)

## v1.2.6

- Fix: The problem that page_list widget doesn't show the picture of revision.author
- Fix: Change implementation of Bootstrap3 toggle switch for admin pages

## v1.2.5

- Feature: crowi-plus Simplified Behavior
- Improvement: Ensure to be able to disable Timeline feature

## v1.2.4

- Fix: Internal Server Error has occurred when a guest user visited the page someone added "liked"

## v1.2.3

- Improvement: Ensure to be able to use Presentation Mode even when not logged in
- Improvement: Presentation Mode on IE11 (Experimental)
- Fix: Broken Presentation Mode

## v1.2.2

- Support: Merge official Crowi (master branch)

## v1.2.1

- Fix: buildIndex error occured when access to installer

## v1.2.0

- Support: Merge official Crowi v1.6.2

## v1.1.12

- Feature: Remove Comment Button

## v1.1.11

- Fix: Omit Comment form from page_list (crowi-plus Enhanced Layout)
- Fix: .search-box is broken on sm/xs screen

## v1.1.10

- Fix: .search-box is broken on sm/xs screen
- Support: Browsable with IE11 (Experimental)

## v1.1.9

- Improvement: Ensure to generate indices of Elasticsearch when installed
- Fix: Specify the version of Bonsai Elasticsearch on Heroku

## v1.1.8

- Fix: Depth of dropdown-menu when `.on-edit`
- Fix: Error occured on saveing with `Ctrl-S`
- Fix: Guest users browsing

## v1.1.7

- Feature: Add option to allow guest users to browse
- Fix: crowi-plus Enhanced Layout

## v1.1.6

- Fix: crowi-plus Enhanced Layout

## v1.1.5

- Fix: crowi-plus Enhanced Layout
- Support: Merge official Crowi v1.6.1 master branch [573144b]

## v1.1.4

- Feature: Ensure to select layout type from Admin Page
- Feature: Add crowi-plus Enhanced Layout

## v1.1.3

- Improvement: Use POSIX-style paths (bollowed crowi/crowi#219 by @Tomasom)

## v1.1.2

- Imprv: Brushup fonts and styles
- Fix: Ensure to specity revision id when saving with `Ctrl-S`

## v1.1.1

- Feature: Save with `Ctrl-S`
- Imprv: Brushup fonts and styles

## v1.1.0

- Support: Merge official Crowi v1.6.1

## v1.0.9

- Feature: Delete user
- Feature: Upload other than images

## v1.0.8

- Feature: Ensure to delete page completely
- Feature: Ensure to delete redirect page
- Fix: https access to Gravatar (this time for sure)

## v1.0.7

- Feature: Keyboard navigation for search box
- Improvement: Intelligent Search

## v1.0.6

- Feature: Copy button that copies page path to clipboard
- Fix: https access to Gravatar
- Fix: server watching crash with `Error: read ECONNRESET` on Google Chrome

## v1.0.5

- Feature: Ensure to use Gravatar for profile image

## v1.0.4

- Improvement: Detach code blocks before preProcess
- Support: Ensure to deploy to Heroku with INSTALL_PLUGINS env
- Support: Ensure to load plugins easily when development

## v1.0.3

- Improvement: Adjust styles

## v1.0.2

- Improvement: For lsx

## v1.0.1

- Feature: Custom CSS
- Support: Notify build failure to Slask

## v1.0.0

- Feature: Plugin mechanism
- Feature: Switchable LineBreaks ON/OFF from admin page
- Improvement: Exclude Environment-dependency
- Improvement: Enhanced linker
- Support: Add Dockerfile
- Support: Abolish gulp
- Support: LiveReload
- Support: Update libs

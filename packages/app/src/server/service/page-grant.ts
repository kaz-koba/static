import { pagePathUtils, pathUtils, pageUtils } from '@growi/core';
import escapeStringRegexp from 'escape-string-regexp';
import mongoose from 'mongoose';

import { IRecordApplicableGrant } from '~/interfaces/page-grant';
import { PageDocument, PageModel } from '~/server/models/page';
import UserGroup from '~/server/models/user-group';
import { isIncludesObjectId, excludeTestIdsFromTargetIds } from '~/server/util/compare-objectId';

const { addTrailingSlash } = pathUtils;
const { isTopPage } = pagePathUtils;

const LIMIT_FOR_MULTIPLE_PAGE_OP = 20;

type ObjectIdLike = mongoose.Types.ObjectId | string;

type ComparableTarget = {
  grant: number,
  grantedUserIds?: ObjectIdLike[],
  grantedGroupId?: ObjectIdLike,
  applicableUserIds?: ObjectIdLike[],
  applicableGroupIds?: ObjectIdLike[],
};

type ComparableAncestor = {
  grant: number,
  grantedUserIds: ObjectIdLike[],
  applicableUserIds?: ObjectIdLike[],
  applicableGroupIds?: ObjectIdLike[],
};

type ComparableDescendants = {
  isPublicExist: boolean,
  grantedUserIds: ObjectIdLike[],
  grantedGroupIds: ObjectIdLike[],
};

class PageGrantService {

  crowi!: any;

  constructor(crowi: any) {
    this.crowi = crowi;
  }

  private validateComparableTarget(comparable: ComparableTarget) {
    const Page = mongoose.model('Page') as unknown as PageModel;

    const { grant, grantedUserIds, grantedGroupId } = comparable;

    if (grant === Page.GRANT_OWNER && (grantedUserIds == null || grantedUserIds.length !== 1)) {
      throw Error('grantedUserIds must not be null and must have 1 length');
    }
    if (grant === Page.GRANT_USER_GROUP && grantedGroupId == null) {
      throw Error('grantedGroupId is not specified');
    }
  }

  /**
   * About the rule of validation, see: https://dev.growi.org/61b2cdabaa330ce7d8152844
   * @returns boolean
   */
  private processValidation(target: ComparableTarget, ancestor: ComparableAncestor, descendants?: ComparableDescendants): boolean {
    this.validateComparableTarget(target);

    const Page = mongoose.model('Page') as unknown as PageModel;

    /*
     * ancestor side
     */
    // GRANT_PUBLIC
    if (ancestor.grant === Page.GRANT_PUBLIC) { // any page can exist under public page
      // do nothing
    }
    // GRANT_OWNER
    else if (ancestor.grant === Page.GRANT_OWNER) {
      if (target.grantedUserIds?.length !== 1) {
        return false;
      }

      if (target.grant !== Page.GRANT_OWNER) { // only GRANT_OWNER page can exist under GRANT_OWNER page
        return false;
      }

      if (ancestor.grantedUserIds[0].toString() !== target.grantedUserIds[0].toString()) { // the grantedUser must be the same as parent's under the GRANT_OWNER page
        return false;
      }
    }
    // GRANT_USER_GROUP
    else if (ancestor.grant === Page.GRANT_USER_GROUP) {
      if (ancestor.applicableGroupIds == null || ancestor.applicableUserIds == null) {
        throw Error('applicableGroupIds and applicableUserIds are not specified');
      }

      if (target.grant === Page.GRANT_PUBLIC) { // public page must not exist under GRANT_USER_GROUP page
        return false;
      }

      if (target.grant === Page.GRANT_OWNER) {
        if (target.grantedUserIds?.length !== 1) {
          throw Error('grantedUserIds must have one user');
        }

        if (!isIncludesObjectId(ancestor.applicableUserIds, target.grantedUserIds[0])) { // GRANT_OWNER pages under GRAND_USER_GROUP page must be owned by the member of the grantedGroup of the GRAND_USER_GROUP page
          return false;
        }
      }

      if (target.grant === Page.GRANT_USER_GROUP) {
        if (target.grantedGroupId == null) {
          throw Error('grantedGroupId must not be null');
        }

        if (!isIncludesObjectId(ancestor.applicableGroupIds, target.grantedGroupId)) { // only child groups or the same group can exist under GRANT_USER_GROUP page
          return false;
        }
      }
    }

    if (descendants == null) {
      return true;
    }
    /*
     * descendant side
     */

    // GRANT_PUBLIC
    if (target.grant === Page.GRANT_PUBLIC) { // any page can exist under public page
      // do nothing
    }
    // GRANT_OWNER
    else if (target.grant === Page.GRANT_OWNER) {
      if (target.grantedUserIds?.length !== 1) {
        throw Error('grantedUserIds must have one user');
      }

      if (descendants.isPublicExist) { // public page must not exist under GRANT_OWNER page
        return false;
      }

      if (descendants.grantedGroupIds.length !== 0 || descendants.grantedUserIds.length > 1) { // groups or more than 2 grantedUsers must not be in descendants
        return false;
      }

      if (descendants.grantedUserIds.length === 1 && descendants.grantedUserIds[0].toString() !== target.grantedUserIds[0].toString()) { // if Only me page exists, then all of them must be owned by the same user as the target page
        return false;
      }
    }
    // GRANT_USER_GROUP
    else if (target.grant === Page.GRANT_USER_GROUP) {
      if (target.applicableGroupIds == null || target.applicableUserIds == null) {
        throw Error('applicableGroupIds and applicableUserIds must not be null');
      }

      if (descendants.isPublicExist) { // public page must not exist under GRANT_USER_GROUP page
        return false;
      }

      const shouldNotExistGroupIds = excludeTestIdsFromTargetIds(descendants.grantedGroupIds, target.applicableGroupIds);
      const shouldNotExistUserIds = excludeTestIdsFromTargetIds(descendants.grantedUserIds, target.applicableUserIds);
      if (shouldNotExistGroupIds.length !== 0 || shouldNotExistUserIds.length !== 0) {
        return false;
      }
    }

    return true;
  }

  /**
   * Prepare ComparableTarget
   * @returns Promise<ComparableAncestor>
   */
  private async generateComparableTarget(
      grant, grantedUserIds: ObjectIdLike[] | undefined, grantedGroupId: ObjectIdLike | undefined, includeApplicable: boolean,
  ): Promise<ComparableTarget> {
    if (includeApplicable) {
      const Page = mongoose.model('Page') as unknown as PageModel;
      const UserGroupRelation = mongoose.model('UserGroupRelation') as any; // TODO: Typescriptize model

      let applicableUserIds: ObjectIdLike[] | undefined;
      let applicableGroupIds: ObjectIdLike[] | undefined;

      if (grant === Page.GRANT_USER_GROUP) {
        const targetUserGroup = await UserGroup.findOne({ _id: grantedGroupId });
        if (targetUserGroup == null) {
          throw Error('Target user group does not exist');
        }

        const relatedUsers = await UserGroupRelation.find({ relatedGroup: targetUserGroup._id });
        applicableUserIds = relatedUsers.map(u => u.relatedUser);

        const applicableGroups = grantedGroupId != null ? await UserGroup.findGroupsWithDescendantsById(grantedGroupId) : null;
        applicableGroupIds = applicableGroups?.map(g => g._id) || null;
      }

      return {
        grant,
        grantedUserIds,
        grantedGroupId,
        applicableUserIds,
        applicableGroupIds,
      };
    }

    return {
      grant,
      grantedUserIds,
      grantedGroupId,
    };
  }

  /**
   * Prepare ComparableAncestor
   * @param targetPath string of the target path
   * @returns Promise<ComparableAncestor>
   */
  private async generateComparableAncestor(targetPath: string, includeNotMigratedPages: boolean): Promise<ComparableAncestor> {
    const Page = mongoose.model('Page') as unknown as PageModel;
    const { PageQueryBuilder } = Page;
    const UserGroupRelation = mongoose.model('UserGroupRelation') as any; // TODO: Typescriptize model

    let applicableUserIds: ObjectIdLike[] | undefined;
    let applicableGroupIds: ObjectIdLike[] | undefined;

    /*
     * make granted users list of ancestor's
     */
    const builderForAncestors = new PageQueryBuilder(Page.find(), false);
    if (!includeNotMigratedPages) {
      builderForAncestors.addConditionAsOnTree();
    }
    const ancestors = await builderForAncestors
      .addConditionToListOnlyAncestors(targetPath)
      .addConditionToSortPagesByDescPath()
      .query
      .exec();
    const testAncestor = ancestors[0]; // TODO: consider when duplicate testAncestors exist
    if (testAncestor == null) {
      throw Error('testAncestor must exist');
    }

    if (testAncestor.grant === Page.GRANT_USER_GROUP) {
      // make a set of all users
      const grantedRelations = await UserGroupRelation.find({ relatedGroup: testAncestor.grantedGroup }, { _id: 0, relatedUser: 1 });
      const grantedGroups = await UserGroup.findGroupsWithDescendantsById(testAncestor.grantedGroup);
      applicableGroupIds = grantedGroups.map(g => g._id);
      applicableUserIds = Array.from(new Set(grantedRelations.map(r => r.relatedUser))) as ObjectIdLike[];
    }

    return {
      grant: testAncestor.grant,
      grantedUserIds: testAncestor.grantedUsers,
      applicableUserIds,
      applicableGroupIds,
    };
  }

  /**
   * Prepare ComparableDescendants
   * @param targetPath string of the target path
   * @returns ComparableDescendants
   */
  private async generateComparableDescendants(targetPath: string, user, includeNotMigratedPages: boolean): Promise<ComparableDescendants> {
    const Page = mongoose.model('Page') as unknown as PageModel;
    const UserGroupRelation = mongoose.model('UserGroupRelation') as any; // TODO: Typescriptize model

    // Build conditions
    const $match: {$or: any} = {
      $or: [],
    };

    const commonCondition = {
      path: new RegExp(`^${escapeStringRegexp(addTrailingSlash(targetPath))}`, 'i'),
      isEmpty: false,
    };

    const conditionForNormalizedPages: any = {
      ...commonCondition,
      parent: { $ne: null },
    };
    $match.$or.push(conditionForNormalizedPages);

    if (includeNotMigratedPages) {
      // Add grantCondition for not normalized pages
      const userGroups = await UserGroupRelation.findAllUserGroupIdsRelatedToUser(user);
      const grantCondition = Page.generateGrantCondition(user, userGroups);
      const conditionForNotNormalizedPages = {
        $and: [
          {
            ...commonCondition,
            parent: null,
          },
          grantCondition,
        ],
      };
      $match.$or.push(conditionForNotNormalizedPages);
    }

    const result = await Page.aggregate([
      { // match to descendants excluding empty pages
        $match,
      },
      {
        $project: {
          _id: 0,
          grant: 1,
          grantedUsers: 1,
          grantedGroup: 1,
        },
      },
      { // remove duplicates from pipeline
        $group: {
          _id: '$grant',
          grantedGroupSet: { $addToSet: '$grantedGroup' },
          grantedUsersSet: { $addToSet: '$grantedUsers' },
        },
      },
      { // flatten granted user set
        $unwind: {
          path: '$grantedUsersSet',
        },
      },
    ]);

    // GRANT_PUBLIC group
    const isPublicExist = result.some(r => r._id === Page.GRANT_PUBLIC);
    // GRANT_OWNER group
    const grantOwnerResult = result.filter(r => r._id === Page.GRANT_OWNER)[0]; // users of GRANT_OWNER
    const grantedUserIds: ObjectIdLike[] = grantOwnerResult?.grantedUsersSet ?? [];
    // GRANT_USER_GROUP group
    const grantUserGroupResult = result.filter(r => r._id === Page.GRANT_USER_GROUP)[0]; // users of GRANT_OWNER
    const grantedGroupIds = grantUserGroupResult?.grantedGroupSet ?? [];

    return {
      isPublicExist,
      grantedUserIds,
      grantedGroupIds,
    };
  }

  /**
   * About the rule of validation, see: https://dev.growi.org/61b2cdabaa330ce7d8152844
   * Only v5 schema pages will be used to compare by default (Set includeNotMigratedPages to true to include v4 schema pages as well).
   * When comparing, it will use path regex to collect pages instead of using parent attribute of the Page model. This is reasonable since
   * using the path attribute is safer than using the parent attribute in this case. 2022.02.13 -- Taichi Masuyama
   * @returns Promise<boolean>
   */
  async isGrantNormalized(
      // eslint-disable-next-line max-len
      user, targetPath: string, grant, grantedUserIds?: ObjectIdLike[], grantedGroupId?: ObjectIdLike, shouldCheckDescendants = false, includeNotMigratedPages = false,
  ): Promise<boolean> {
    if (isTopPage(targetPath)) {
      return true;
    }

    const comparableAncestor = await this.generateComparableAncestor(targetPath, includeNotMigratedPages);

    if (!shouldCheckDescendants) { // checking the parent is enough
      const comparableTarget = await this.generateComparableTarget(grant, grantedUserIds, grantedGroupId, false);
      return this.processValidation(comparableTarget, comparableAncestor);
    }

    const comparableTarget = await this.generateComparableTarget(grant, grantedUserIds, grantedGroupId, true);
    const comparableDescendants = await this.generateComparableDescendants(targetPath, user, includeNotMigratedPages);

    return this.processValidation(comparableTarget, comparableAncestor, comparableDescendants);
  }

  /**
   * Separate normalizable pages and NOT normalizable pages by PageService.prototype.isGrantNormalized method.
   * normalizable pages = Pages which are able to run normalizeParentRecursively method (grant & userGroup rule is correct)
   * @param pageIds pageIds to be tested
   * @returns a tuple with the first element of normalizable pages and the second element of NOT normalizable pages
   */
  async separateNormalizableAndNotNormalizablePages(user, pages): Promise<[(PageDocument & { _id: any })[], (PageDocument & { _id: any })[]]> {
    if (pages.length > LIMIT_FOR_MULTIPLE_PAGE_OP) {
      throw Error(`The maximum number of pageIds allowed is ${LIMIT_FOR_MULTIPLE_PAGE_OP}.`);
    }

    const shouldCheckDescendants = true;
    const shouldIncludeNotMigratedPages = true;

    const normalizable: (PageDocument & { _id: any })[] = [];
    const nonNormalizable: (PageDocument & { _id: any })[] = []; // can be used to tell user which page failed to migrate

    for await (const page of pages) {
      const {
        path, grant, grantedUsers: grantedUserIds, grantedGroup: grantedGroupId,
      } = page;

      if (!pageUtils.isPageNormalized(page)) {
        nonNormalizable.push(page);
        continue;
      }

      if (await this.isGrantNormalized(user, path, grant, grantedUserIds, grantedGroupId, shouldCheckDescendants, shouldIncludeNotMigratedPages)) {
        normalizable.push(page);
      }
      else {
        nonNormalizable.push(page);
      }
    }

    return [normalizable, nonNormalizable];
  }

  async calcApplicableGrantData(page, user): Promise<IRecordApplicableGrant> {
    const Page = mongoose.model('Page') as unknown as PageModel;
    const UserGroupRelation = mongoose.model('UserGroupRelation') as any; // TODO: Typescriptize model

    // -- Public only if top page
    const isOnlyPublicApplicable = isTopPage(page.path);
    if (isOnlyPublicApplicable) {
      return {
        [Page.GRANT_PUBLIC]: null,
      };
    }

    // Increment an object (type IRecordApplicableGrant)
    // grant is never public, anyone with the link, nor specified
    const data: IRecordApplicableGrant = {
      [Page.GRANT_RESTRICTED]: null, // any page can be restricted
    };

    // -- Any grant is allowed if parent is null
    const isAnyGrantApplicable = page.parent == null;
    if (isAnyGrantApplicable) {
      data[Page.GRANT_PUBLIC] = null;
      data[Page.GRANT_OWNER] = null;
      data[Page.GRANT_USER_GROUP] = await UserGroupRelation.findAllUserGroupIdsRelatedToUser(user);
      return data;
    }

    const parent = await Page.findById(page.parent);
    if (parent == null) {
      throw Error('The page\'s parent does not exist.');
    }

    const {
      grant, grantedUsers, grantedGroup,
    } = parent;

    if (grant === Page.GRANT_PUBLIC) {
      data[Page.GRANT_PUBLIC] = null;
      data[Page.GRANT_OWNER] = null;
      data[Page.GRANT_USER_GROUP] = await UserGroupRelation.findAllUserGroupIdsRelatedToUser(user);
    }
    else if (grant === Page.GRANT_OWNER) {
      const grantedUser = grantedUsers[0];

      const isUserApplicable = grantedUser.toString() === user._id.toString();

      if (isUserApplicable) {
        data[Page.GRANT_OWNER] = null;
      }
    }
    else if (grant === Page.GRANT_USER_GROUP) {
      const group = await UserGroup.findById(grantedGroup);
      if (group == null) {
        throw Error('Group not found to calculate grant data.');
      }

      const applicableGroups = await UserGroupRelation.findGroupsWithDescendantsByGroupAndUser(group, user);

      const isUserExistInGroup = await UserGroupRelation.countByGroupIdAndUser(group, user) > 0;

      if (isUserExistInGroup) {
        data[Page.GRANT_OWNER] = null;
      }
      data[Page.GRANT_USER_GROUP] = { applicableGroups };
    }

    return data;
  }

}

export default PageGrantService;

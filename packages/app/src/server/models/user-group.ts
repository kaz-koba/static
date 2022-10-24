import mongoose, {
  Schema, Model, Document,
} from 'mongoose';
import mongoosePaginate from 'mongoose-paginate-v2';

import { IUserGroup } from '~/interfaces/user';

import { getOrCreateModel } from '../util/mongoose-utils';


export interface UserGroupDocument extends IUserGroup, Document {}

export interface UserGroupModel extends Model<UserGroupDocument> {
  [x:string]: any, // for old methods

  PAGE_ITEMS: 10,
}

/*
 * define schema
 */
const ObjectId = mongoose.Schema.Types.ObjectId;

const schema = new Schema<UserGroupDocument, UserGroupModel>({
  name: { type: String, required: true, unique: true },
  parent: { type: ObjectId, ref: 'UserGroup', index: true },
  description: { type: String, default: '' },
}, {
  timestamps: true,
});
schema.plugin(mongoosePaginate);

const PAGE_ITEMS = 10;

schema.statics.findUserGroupsWithPagination = function(opts) {
  const query = { parent: null };
  const options = Object.assign({}, opts);
  if (options.page == null) {
    options.page = 1;
  }
  if (options.limit == null) {
    options.limit = PAGE_ITEMS;
  }

  return this.paginate(query, options)
    .catch((err) => {
      // debug('Error on pagination:', err); TODO: add logger
    });
};


schema.statics.findChildUserGroupsByParentIds = async function(parentIds, includeGrandChildren = false) {
  if (!Array.isArray(parentIds)) {
    throw Error('parentIds must be an array.');
  }

  const childUserGroups = await this.find({ parent: { $in: parentIds } });

  let grandChildUserGroups: UserGroupDocument[] | null = null;
  if (includeGrandChildren) {
    const childUserGroupIds = childUserGroups.map(group => group._id);
    grandChildUserGroups = await this.find({ parent: { $in: childUserGroupIds } });
  }

  return {
    childUserGroups,
    grandChildUserGroups,
  };
};

schema.statics.countUserGroups = function() {
  return this.estimatedDocumentCount();
};

schema.statics.createGroup = async function(name, description, parentId) {
  // create without parent
  if (parentId == null) {
    return this.create({ name, description });
  }

  // create with parent
  const parent = await this.findOne({ _id: parentId });
  if (parent == null) {
    throw Error('Parent does not exist.');
  }
  return this.create({ name, description, parent });
};

/**
 * Find all ancestor groups starting from the UserGroup of the initial "group".
 * Set "ancestors" as "[]" if the initial group is unnecessary as result.
 * @param groups UserGroupDocument
 * @param ancestors UserGroupDocument[]
 * @returns UserGroupDocument[]
 */
schema.statics.findGroupsWithAncestorsRecursively = async function(group, ancestors = [group]) {
  if (group == null) {
    return ancestors;
  }

  const parent = await this.findOne({ _id: group.parent });
  if (parent == null) {
    return ancestors;
  }

  ancestors.unshift(parent);

  return this.findGroupsWithAncestorsRecursively(parent, ancestors);
};

/**
 * Find all descendant groups starting from the UserGroups in the initial groups in "groups".
 * Set "descendants" as "[]" if the initial groups are unnecessary as result.
 * @param groups UserGroupDocument[] including at least one UserGroup
 * @param descendants UserGroupDocument[]
 * @returns UserGroupDocument[]
 */
schema.statics.findGroupsWithDescendantsRecursively = async function(groups, descendants = groups) {
  const nextGroups = await this.find({ parent: { $in: groups.map(g => g._id) } });

  if (nextGroups.length === 0) {
    return descendants;
  }

  return this.findGroupsWithDescendantsRecursively(nextGroups, descendants.concat(nextGroups));
};

schema.statics.findGroupsWithDescendantsById = async function(groupId) {
  const root = await this.findOne({ _id: groupId });
  if (root == null) {
    throw Error('The root user group does not exist');
  }
  return this.findGroupsWithDescendantsRecursively([root]);
};

export default getOrCreateModel<UserGroupDocument, UserGroupModel>('UserGroup', schema);

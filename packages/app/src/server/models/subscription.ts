import {
  SubscriptionStatusType, AllSubscriptionStatusType, Ref, IPage, IUser, ISubscription,
} from '@growi/core';
import {
  Types, Document, Model, Schema,
} from 'mongoose';

import { AllSupportedTargetModels } from '~/interfaces/activity';

import { getOrCreateModel } from '../util/mongoose-utils';


export interface SubscriptionDocument extends ISubscription, Document {}

export interface SubscriptionModel extends Model<SubscriptionDocument> {
  findByUserIdAndTargetId(userId: Types.ObjectId | string, targetId: Types.ObjectId | string): any
  upsertSubscription(user: Ref<IUser>, targetModel: string, target: Ref<IPage>, status: string): any
  subscribeByPageId(userId: Types.ObjectId, pageId: Types.ObjectId, status: string): any
  getSubscription(target: Ref<IPage>): Promise<Ref<IUser>[]>
  getUnsubscription(target: Ref<IPage>): Promise<Ref<IUser>[]>
  getSubscriptions(targets: Ref<IPage>[]): Promise<Ref<IUser>[]>
}

const subscriptionSchema = new Schema<SubscriptionDocument, SubscriptionModel>({
  user: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    index: true,
    required: true,
  },
  targetModel: {
    type: String,
    required: true,
    enum: AllSupportedTargetModels,
  },
  target: {
    type: Schema.Types.ObjectId,
    ref: 'Page',
    refPath: 'targetModel',
    required: true,
  },
  status: {
    type: String,
    required: true,
    enum: AllSubscriptionStatusType,
  },
}, {
  timestamps: true,
});

subscriptionSchema.methods.isSubscribing = function() {
  return this.status === SubscriptionStatusType.SUBSCRIBE;
};

subscriptionSchema.methods.isUnsubscribing = function() {
  return this.status === SubscriptionStatusType.UNSUBSCRIBE;
};

subscriptionSchema.statics.findByUserIdAndTargetId = function(userId, targetId) {
  return this.findOne({ user: userId, target: targetId });
};

subscriptionSchema.statics.upsertSubscription = function(user, targetModel, target, status) {
  const query = { user, targetModel, target };
  const doc = { ...query, status };
  const options = {
    upsert: true, new: true, setDefaultsOnInsert: true, runValidators: true,
  };
  return this.findOneAndUpdate(query, doc, options);
};

subscriptionSchema.statics.subscribeByPageId = function(userId, pageId, status) {
  return this.upsertSubscription(userId, 'Page', pageId, status);
};

subscriptionSchema.statics.getSubscription = async function(target: Ref<IPage>) {
  return this.find({ target, status: SubscriptionStatusType.SUBSCRIBE }).distinct('user');
};

subscriptionSchema.statics.getUnsubscription = async function(target: Ref<IPage>) {
  return this.find({ target, status: SubscriptionStatusType.UNSUBSCRIBE }).distinct('user');
};

subscriptionSchema.statics.getSubscriptions = async function(targets: Ref<IPage>[]) {
  return this.find({ target: { $in: targets }, status: SubscriptionStatusType.SUBSCRIBE }).distinct('user');
};

export default getOrCreateModel<SubscriptionDocument, SubscriptionModel>('Subscription', subscriptionSchema);

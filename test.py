def per_minibatch(sample_batch, base, rpn, detector, cls_loss_criteria, reg_loss_criteria, detect_loss_criteria, optimizer, anchor):
    image, mask, ground_box, label = sample_batch['image'], sample_batch['mask'], sample_batch['box'], sample_batch['label'].squeeze(1)
    # assuming sample_batch is of size 100*3*1*1, change to 100*6*6*3
    ground_box = ground_box.repeat(1, 1, 6, 6)    
    image, mask, box, label = Variable(image), Variable(mask), Variable(ground_box), Variable(label)

    if torch.cuda.is_available():
        image = image.cuda()
        mask = mask.cuda()
        box = box.cuda()
        label = label.cuda()

    region_logical = torch.lt(mask, 2)
    reg_region_logical = torch.eq(mask, 1)    
    mask_sub = torch.masked_select(mask, region_logical)    
    region = region_logical.float()

    base_conv = base(image)
    cls_feat, reg_feat = rpn(base_conv)
    cls_feat = cls_feat.squeeze(1) # BATCH_SIZE*CHANNEL*H*W

    #zero out all irrelevant positions
    cls_feat = cls_feat * region 
    mask = mask.float() * region
    # -- find maximally activated bouding box
    cls_feat = cls_feat.view(100, -1)
    cls_feat_max, cls_argmax = torch.max(cls_feat, 1)
    reg_feat_reshape = reg_feat.view(100, 3, -1)
    
    reg_x = torch.diag(torch.index_select(reg_feat_reshape[:, 0, :], 1, cls_argmax))
    reg_y = torch.diag(torch.index_select(reg_feat_reshape[:, 1, :], 1, cls_argmax))
    reg_w = torch.diag(torch.index_select(reg_feat_reshape[:, 2, :], 1, cls_argmax))
    
    reg_feat_singl = torch.stack((reg_x, reg_y, reg_w), 1)
    # -- use it to get theta parameters
    
    theta = get_theta(reg_feat_singl)
    output_stn = stn_transform(theta, base_conv, torch.Size((BATCH_SIZE, 256, 4, 4)))
    
    # -- stn complete
    class_preds = detector(output_stn)

    _, predicted = torch.max(class_preds.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum()
    
    return correct, total



def test_minibatch(sample_batch, base, rpn, reg_loss_func, anchor):
    image, mask, ground_box, img_names = sample_batch['image'], sample_batch['mask'], sample_batch['box'], sample_batch['name']

    ground_box = ground_box.repeat(1, 1, 6, 6)  
    image, mask, box = Variable(image), Variable(mask), Variable(ground_box)
    
    if torch.cuda.is_available():
        image = image.cuda()
        mask = mask.cuda()
        box = box.cuda()
    
    region_logical = torch.lt(mask, 2)
    reg_region_logical = torch.eq(mask, 1)

    region = region_logical.float()
    num = torch.sum(region).data[0]
    num_reg = torch.sum(reg_region_logical.float()).data[0]

    mask_sub = torch.masked_select(mask, region_logical)
    
    base_conv = base(image)
    cls_feat, reg_feat = rpn(base_conv)
    # -- find maximally activated bouding box
    """
    cls_feat = cls_feat.squeeze(1) # BATCH_SIZE*CHANNEL*H*W
    max_p = nn.MaxPool2d(6,6)
    cls_feat_max = max_p(cls_feat)
    cls_argmax = cls_feat == cls_feat_max
    cls_argmax = cls_argmax.unsqueeze(1)
    cls_argmax = cls_argmax.repeat(1,3,1,1)
    reg_feat_singl = torch.masked_select(reg_feat, cls_argmax).view(BATCH_SIZE, 3)    
    # -- use it to get theta parameters
    
    theta = get_theta(reg_feat_singl)
    plot_image(image[0, :, :, :].data.cpu())
    output_stn = stn_transform(theta, image, torch.Size((BATCH_SIZE, 3, 32, 32)))
    plot_image(output_stn[0, :, :, :].data.cpu())
    """
    # -- stn complete
    plot_image(image[0, :, :, :].data.cpu())
    
    cls_feat = cls_feat.view(100, -1)
    cls_feat_max, cls_argmax = torch.max(cls_feat, 1)
    reg_feat_reshape = reg_feat.view(100, 3, -1)
    
    reg_x = torch.diag(torch.index_select(reg_feat_reshape[:, 0, :], 1, cls_argmax))
    reg_y = torch.diag(torch.index_select(reg_feat_reshape[:, 1, :], 1, cls_argmax))
    reg_w = torch.diag(torch.index_select(reg_feat_reshape[:, 2, :], 1, cls_argmax))
    
    reg_feat_singl = torch.stack((reg_x, reg_y, reg_w), 1)
    # -- use it to get theta parameters
    
    theta = get_theta(reg_feat_singl)
    output_stn = stn_transform(theta, base_conv, torch.Size((BATCH_SIZE, 256, 4, 4)))

    cls_feat = cls_feat.view(BATCH_SIZE, 6, 6)
    preds = cls_feat > 0.5
    
    preds_sub = torch.masked_select(preds, region_logical) #selects and converts to a 1D vector
    acc = torch.sum(torch.eq(mask_sub, preds_sub).float()).data[0]*1.0 / num
    rbox_loss = get_reg_loss(reg_loss_func, box, reg_feat, anchor, reg_region_logical.float())
    
    return acc, rbox_loss.data[0]/num_reg
    
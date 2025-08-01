methods=["FLOAT-16","INT-8","CE_AdaSAC_ALPHA_0.9","CE_AdaSAC_ALPHA_0.1","only_split"]
model=['GPTJ-6B',"Resnet50","VGG-16","AlexNet"]
loss_gpt = [1.9674,'2.2101',2.08253515625,3.4496484375,1.9674]
loss_resnet50 = [0.9681,0.9682,0.9678,1.0641642015576362,'','',0.9681]
loss_vgg16 = [1.1538,1.1538,1.1534,'','',1.1538]
loss_alex = [1.9227,1.9226,1.9223,'','',1.9227]
overhead_on_edge_gptj=[11276640312,11276640312,8813297692,7323779100,9396731932]
overhead_decress_gptj=[1,1,0.781,0.6494,0.8332]
overhead_increas_res50=[8267485184,8267485184,2647900000,'','',]
overhead_increas_vgg16=[31049009152,31049009152,'','','',]
overhead_increas_alex=[1428413824,1428413824,'','','',]


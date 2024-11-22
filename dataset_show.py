import torch
from torch.utils.data import Dataset
import numpy as np
import os

import cv2
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes # 是一个字典
from nuscenes.utils.data_classes import Box

class nuScenesDataSet(Dataset):
    def __init__(self, root="/root/autodl-tmp/data/nuScenes", mode="v1.0-trainval"):# trainval
        super(nuScenesDataSet,self).__init__()
        
        self.version = mode
        self.dataroot = root
        self.nusc = NuScenes(self.version,self.dataroot,verbose=False)
        self.nusc_exp = NuScenesExplorer(self.nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        
        self.temporalsize = 10 # 突然觉得处理3个时序很难，还是先处理1个时序，只是作简单的识别吧
        self.category_color,self.category_label = self.get_classlabel()
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = {}
        keys = ['image', 'ego_to_img', 
                'image_canny',
                'img_labels',# 可以像YOLO-6D一样，预测点,突然有一个想法，我要用canny数据先生成一些图像平面的角点
                'ego_lable',
                'depths',
                'segmentation'
                ]
        for key in keys:
            data[key] = []
            
        segmentation = []
        image = []
        image_canny = []
        ego_to_img = []
        img_labels = []
        ego_lable = []
        depths = []
        for i, index_token in enumerate(self.indices[idx]):
            print(i, index_token)
            self.index = index_token
            samp = self.nusc.get("sample", index_token)
            """下面都是这个token时间戳的数据"""
            
            '''怎么逻辑这么混乱
            1、雷达信息有最原始的雷达点,有雷达对齐到车身,车身对齐到世界坐标系'''
            globel_points,lidar_points,global_to_ego,lidar_img = self.get_lidar_data(samp)
            '''
            2、标注数据是global的角点,中心,名称'''
            global_corners,global_centers,category_nameAN,lidar_category_label,ego_lables  = self.get_annotation(samp,global_to_ego,lidar_img)# 三维框的角点是四维座标点
            segmentation.append(lidar_category_label)
            # 图像数据有6个真烦人,内参和外参还要对应起来!   那么外参我直接把cam对到世界坐标吧
            '''gt_labelsN是图像上的框的位置'''
            '''
            3、相机数据包括图像,相机到图像,相机到车身,车身到世界'''
            imgN,cannyN,ego_to_imgN,globel_to_imgN,img_labelsN = self.get_camera_data(samp,global_corners,global_centers,category_nameAN)
            image.append(imgN)
            image_canny.append(cannyN)
            ego_to_img.append(ego_to_imgN)
            img_labels.append(img_labelsN)
            ego_lable.append(ego_lables)
            """内参:camera_to_img 是intrinsic
            外参:camera_to_global 就直接是 extrinsicN"""  
            # 然后 深度图的深度都是针对 相机坐标系的,采用一维黑白值表达就行,越近越小,看起来越亮(0-255)
            # 就循环6次完成吧
            # deep_imgN = self.get_depth_data(globel_points,globel_to_imgN)
            # depths.append(deep_imgN)
            # print(len(imgN),len(extrinsicN),len(intrinsicN),len(deep_imgN)) # 6 6 6 6 
            # (900, 1600, 3) (4, 4) (4, 4) (900, 1600, 1) 哇，搞不懂，这么大的数据，这么高精确度真的有用吗？
            # /root/autodl-tmp/_Pytorch/B_NuScenes_SHOW/(imgN[0].shape,extrinsicN[0].shape,intrinsicN[0].shape,deep_imgN[0].shape) # <class 'numpy.ndarray'>
            
            """下面处理物体,原始数据都是世界坐标系呀,是要处理成相对车身的吗？
            但是之前处理的鸟瞰图的分割,我不喜欢，因为范围太大了,有必要吗,根本看不到,还检测个濞
            那到底是类别还是实例呢?我觉得把静止物体表一类;车辆 行人
            实例分割吧,行人也各种各样,但是有些确实不重要呀"""
            
            """都统一到雷达采集时候的车身坐标系，那相机数据怎么搞？
            也是转换到lidar—ego吗?
            globel_to_img = intrinsic @ ego_to_camera @ globel_to_ego"""
            
            """----------------------------------------------------------------------------------"""
            # 先不管其他的，最直接的检测，就是给imgN，出global_corners
            # label是和data batch一起的，
        data['segmentation'] = segmentation
        data['image'] = image
        data['image_canny'] = image_canny
        data['ego_to_img'] = ego_to_img
        data['img_labels'] = img_labels
        data['ego_lable'] = ego_lable
        data['depths'] = depths
        
        return data
    
    def get_classlabel(self):
        # 18个类别的独热编码
        # 定义所有类别
        categories = ['human.pedestrian.adult', 
                      'human.pedestrian.child', 
                      'human.pedestrian.construction_worker', 
                      'human.pedestrian.personal_mobility',
                      'human.pedestrian.police_officer', 
                      'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair',# 7
                      'movable_object.barrier', 
                      'movable_object.debris', 
                      'movable_object.pushable_pullable',
                      'movable_object.trafficcone', # 4
                      'static_object.bicycle_rack', 
                      'vehicle.bicycle', 
                      'vehicle.bus.bendy',
                      'vehicle.bus.rigid', 
                      'vehicle.car', 
                      'vehicle.emergency.police',
                      'vehicle.emergency.ambulance',
                      'vehicle.construction', 
                      'vehicle.motorcycle', 
                      'vehicle.trailer', 
                      'vehicle.truck',# 10
                      'animal']
        category_color = [[255, 0, 0],
                            [255, 0, 36],
                            [255, 0, 72],
                            [255, 0, 108], 
                            [255, 0, 144], 
                            [255, 0, 180],
                            [255,0, 216],
                            [128, 0, 252],
                            [128, 0, 189],
                            [128, 0, 128],
                            [128, 0, 63],
                            [128, 128, 128],
                            [0, 255, 255],
                            [0, 200, 255],
                            [0, 175, 255],
                            [0, 150, 255],
                            [0, 125, 255],
                            [0, 100, 255],
                            [0, 75, 255],
                            [0, 50, 255], 
                            [0, 25, 255],
                            [0, 0, 255],
                            [0, 0, 0]]
        
        # 生成独热编码
        one_hot_encoding = np.eye(len(categories))
        # 创建空字典，用于存储类别和对应的独热编码
        class_one_hot = {}
        class_map = {}
        # 将类别和独热编码存入字典
        for i in range(len(categories)):
            class_one_hot[categories[i]] = one_hot_encoding[i]
            class_map[categories[i]] = category_color[i]
  
        return class_map,class_one_hot
    
    def get_indices(self):
        data_list = []
        """
        这里是把所有的数据帧分组,temporalsize是要学的时序的长度
        """
        for sces in self.nusc.scene:
            # sce.keys()
            sce = self.nusc.get("scene", sces['token'])
            # print(sce['name'],sce['nbr_samples'])
            samp = self.nusc.get("sample", sce['first_sample_token'])
            # 这是这个场景的第一个数据帧
            for s in range(sce['nbr_samples']-self.temporalsize):
                tmp_samp = samp
                token_list = []
                for t in range(self.temporalsize):
                    token_list.append(tmp_samp["token"])
                    tmp_samp = self.nusc.get("sample", tmp_samp['next'])
                data_list.append(token_list)
                samp = self.nusc.get("sample", samp['next'])
        # 保存ids
        # data_token_batch = np.array(data_list).reshape(-1,self.temporalsize)
        # np.savetxt('data_token_batch.txt',data_token_batch,delimiter=' ',fmt='%s')
        return np.array(data_list).reshape(-1,self.temporalsize)
    
    def get_camera_data(self, samp,global_corners,global_centers,names):
        intrinsic = np.eye(4)
        imgN = []
        cannyN = []
        # camera_to_globalN = []
        # camera_to_imgN = []
        ego_to_imgN = []
        globel_to_imgN = []
        img_labelsN = []
        
        result_cann = np.zeros((900 * 2, 1600 * 3), dtype=np.uint8)
        result_colo = np.zeros((900 * 2, 1600 * 3, 3), dtype=np.uint8)
        
        for v,cam in enumerate(self.cams):
            camera_token = samp['data'][cam]
            camera_data = self.nusc.get('sample_data',camera_token)

            camera_ego_pose = self.nusc.get('ego_pose',camera_data['ego_pose_token'])# translation rotation
            camera_calibrated_data = self.nusc.get("calibrated_sensor",camera_data['calibrated_sensor_token'])# translation rotation
            intrinsic[:3,:3] = camera_calibrated_data['camera_intrinsic']# 相机内参，可以将雷达点达到图像平面上
            # camera_to_imgN.append(intrinsic)
            """要做环视图的检测，车辆运动还重要吗？其实重要，因为有些静止物体也会动起来呀
            我觉得可以把静止的物体归为一类，单独识别"""
            ego_to_globel = self.get_matrix(camera_ego_pose)# 这是此时相机拍照时候的，车身运动
            camera_to_ego = self.get_matrix(camera_calibrated_data)# 这是相机相对车身的标定
            camera_to_global = ego_to_globel @ camera_to_ego
            # camera_to_globalN.append(camera_to_global)
            """这里再处理一下,需要ego-to-img,我想想,对齐到ego应该没有问题"""
            ego_to_img = intrinsic @ np.linalg.inv(camera_to_ego)
            ego_to_imgN.append(ego_to_img)
            globel_to_img = intrinsic @ np.linalg.inv(camera_to_global)# np.linalg.inv(camera_to_ego) @ np.linalg.inv(ego_to_globel)
            globel_to_imgN.append(globel_to_img)
            # 我始终觉得，这是表达相机位置的参数，感觉其实每一个batch都一样,事实证明内参数确实一样，外参数不一样，可能每一次有抖动吧
            """这里面感觉要考虑下到底用什么读取，读取出来的形式是矩阵吗?
            在Python中使用Open CV读取一张图片后,会保存为“numpy.ndarray”格式,"""
            camera_file = os.path.join(self.dataroot,camera_data['filename'])
            img = cv2.imread(camera_file)
            # hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            
            canny_img = cv2.Canny(image=img, threshold1=0, threshold2=200)
            # cv2.imwrite("canny_img.jpg",canny_img) #暂时没有错误
            # imgN.append(cv2.merge([hsv_img, canny_img]))
            # imgN.append(hsv_img)
            cannyN.append(canny_img)
            
            img_labels = []
            """这一部分是变换到图像上,是三维检测框在图像上的坐标"""
            for globel_corner,global_center,name in zip(global_corners,global_centers,names):
                img_coners = globel_corner @ globel_to_img.T
                img_coners[:,:2] /= img_coners[:,[2]]
                img_coners = img_coners.astype(np.int32) 
                """要把coners的形式搞清楚,有八个点,[U,V,D,1]??是D?""" 
                img_centers = global_center @ globel_to_img.T
                img_centers[:2] /= img_centers[2]
                img_centers = img_centers.astype(np.int32)

                # if any(img_coners[:,[2]]>0):
                    

                if img_centers[2] > 0:
                    # 可视化数据
                    img_labels.append([self.category_label[name],img_centers,img_coners]) # 类别独热编码，中心，角点
                    ix,iy = [0,1,2,3,0,1,2,3,4,5,6,7], \
                            [4,5,6,7,1,2,3,0,5,6,7,4]
                    for p0,p1 in zip(img_coners[ix], img_coners[iy]):
                        if p0[2] <= 0 or p1[2] <=0:continue
                        cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), self.category_color[name], 2, 16)
                        cv2.line(canny_img, (p0[0], p0[1]), (p1[0], p1[1]), 255, 2, 16) 
                    x, y = img_centers[:2]  # 取出前两个元素
                    cv2.circle(img,(int(x), int(y)),10,self.category_color[name], -1)
                    cv2.circle(canny_img,(int(x), int(y)),5,255, -1)    
            
            # 可视化数据
            row = v // 3
            col = v % 3
            # # 将图像放置在大画布的指定位置
            result_cann[row * img.shape[0]:(row + 1) * img.shape[0], 
            col * img.shape[1]:(col + 1) * img.shape[1]] = canny_img
            result_colo[row * img.shape[0]:(row + 1) * img.shape[0], 
            col * img.shape[1]:(col + 1) * img.shape[1]] = img
            
            img_labelsN.append(img_labels)
        # cv2.imwrite("/root/autodl-tmp/_Pytorch/B_NuScenes_SHOW/canny.jpg",result_cann)
        name = os.path.join("/root/autodl-tmp/_Pytorch/B_NuScenes_SHOW", "img.jpg")   
        cv2.imwrite(name,result_colo)
            
        return imgN,cannyN,ego_to_imgN,globel_to_imgN,img_labelsN
            
    def get_lidar_data(self,samp):
        lidar_token = samp['data']['LIDAR_TOP']
        lidar_sample_data = self.nusc.get('sample_data',lidar_token)
        
        lidar_ego_pose = self.nusc.get("ego_pose",lidar_sample_data['ego_pose_token'])
        lidar_calibrated_data = self.nusc.get("calibrated_sensor",lidar_sample_data['calibrated_sensor_token'])
        ego_to_global = self.get_matrix(lidar_ego_pose)
        lidar_to_ego = self.get_matrix(lidar_calibrated_data)
        lidar_to_global = ego_to_global @ lidar_to_ego
        
        """这里的变换逻辑,感觉不用ego pose了,哦,不是，三维点的框 是世界座标点,那预测的时候要换到车身坐标系吧!"""
        lidar_file = os.path.join(self.dataroot,lidar_sample_data['filename'])
        lidar_points = np.fromfile(lidar_file,dtype=np.float32).reshape(-1,5) # 这是作原始的点云数据，【X，Y，Z，I，M】
        hom_points = np.concatenate([lidar_points[:,:3],np.ones((len(lidar_points),1))],axis=1) #这是相对雷达的点幼，毕竟是雷达采集的原始数据
        """可视化雷达点"""
        ego_points = hom_points @ lidar_to_ego.T
        x,y,z = ego_points[:,:3].T
        x = np.round((x +125 ) /0.5) # 为什么要除100，要缩小吗？？？？
        y = np.round((y +125 ) /0.5)
        z = (z-z.min())/(z.min()-z.max())
        lidar_img = np.ones((500,500,3), np.uint8)* 255  # np.zeros((500,500,3), np.uint8)
        for xi,yi,d in zip(x,y,z):
            lidar_img[int(xi),int(yi)] = d*255
        """换到世界坐标系,之后运用相机内外参数，转换到相机平面，求深度图
        不用globel_point,直接就是ego——point不行嘛？
        """
        globel_points = hom_points @ lidar_to_global.T
        ego_points = hom_points @ lidar_to_ego.T
        lidar_point = hom_points
        
        return globel_points,lidar_point,np.linalg.inv(ego_to_global),lidar_img
        
    def get_annotation(self,samp,global_to_ego,lidar_img):
        global_cornersAN = []
        global_centersAN = []
        category_nameAN = []
        ego_lables = []
        # lidar_img = np.ones((500, 500, 3), dtype=np.uint8) * 255 # 创建一个白色的三通道画布     
        for annotation_token in samp['anns']:
            annotation = self.nusc.get("sample_annotation",annotation_token)
            
            #  print(annotation['category_name'])
            name = annotation['category_name']
            category_nameAN.append(name)
            
            box = Box(annotation['translation'],annotation['size'],Quaternion(annotation['rotation']))
            corners = box.corners().T # 这是框的角点，其实我也没有搞清楚，这些点的顺序是怎么形容的  
            global_corners = np.concatenate([corners, np.ones((len(corners), 1))],axis=1) # 是世界标定框,到时候预测要预测相对车身的
            center = box.center# 框的中心
            global_center = np.append(center, 1)

            global_centersAN.append(global_center)
            global_cornersAN.append(global_corners)
            # 这个框的坐标点到底要不要那？我任务还没想好

            """这一部分是变换到 lidar上"""
            ego_coners = global_corners @ global_to_ego.T
            pts = ego_coners[[1,0,4,5],:2]
            pts[:, [1, 0]] = pts[:, [0, 1]]
            pts = np.round((pts +125 ) /0.5).astype(np.int32)
            """既然模型里面是对齐到雷达坐标系上吧,总感觉ego也有偏移"""
            ego_center = global_center @ global_to_ego.T
            ego_lables.append([ego_coners,ego_center])
            cv2.fillPoly(lidar_img, [pts], self.category_color[name])
            cv2.circle(lidar_img,(250,250),10,[255,0,0], 2)        
        name = os.path.join("/root/autodl-tmp/_Pytorch/B_NuScenes_SHOW", "category.jpg")   
        cv2.imwrite(name,lidar_img)
            
        return global_cornersAN,global_centersAN,category_nameAN,lidar_img,ego_lables     
            
    def get_depth_data(self,globel_points,globel_to_imgN):
        """"这里完成雷达数据 依靠相机内外参数的投影，投影到相机平面上
        是一种粗糙的深度方式
        此时此刻的ego是雷达采集时候的ego的点
        而ego-to-img应该是一直不变的,
        """
        result_depth = np.zeros((900 * 2, 1600 * 3, 3), dtype=np.uint8)
        depths = []
        for n in range(6):
            deep_img = np.ones((900, 1600, 1), dtype=np.uint8) * 255
            img_points = globel_points @ globel_to_imgN[n].T # np.linalg.inv(extrinsicN[n]).T @ intrinsicN[n].T
            img_points[:,:2] /= img_points[:,[2]]# 归一化，变成齐次坐标
            img_points = img_points[img_points[:,2] >0, :3]# [U,V,D,1]
            img_points = img_points[np.logical_and(np.logical_and(img_points[:, 0] < 1600, img_points[:, 0] > 0), 
                                    np.logical_and(img_points[:, 1] > 0, img_points[:, 1] < 900)), :3]#并且要筛选可以落到图像上的点,有些看不到
            depth = img_points[:,2] # 相对深度，对于相机而言的
            depth_color = self.depth_gray(depth)
            for p,d in zip(img_points[:,:].astype(int), depth_color):
                # print(d)
                x,y,z = p
                if x<1600 and y<900 and x>0 and y>0:
                    center_x = x
                    center_y = y
                    # 定义正方形的边长和颜色
                    square_size = 10
                    # 计算正方形的边界框坐标
                    x1 = center_x - square_size // 2
                    y1 = center_y - square_size*3 // 2
                    x2 = center_x + square_size // 2
                    y2 = center_y + square_size*3 // 2
                    # 在图像上画正方形
                    cv2.rectangle(deep_img, (x1, y1), (x2, y2), d, thickness=-1)
                # 可视化数据
                row = n // 3
                col = n % 3
                # # 将图像放置在大画布的指定位置
                result_depth[row * deep_img.shape[0]:(row + 1) * deep_img.shape[0], 
                col * deep_img.shape[1]:(col + 1) * deep_img.shape[1]] = deep_img
        cv2.imwrite("deep.jpg",result_depth) #暂时没有错误
        depths.append(deep_img) 
        return depths   
    
    
    def get_matrix(self,calibrated_data):
        """这里就是用一个4*4的矩阵,是方便运算的变换矩阵,给的标定数据"""
        output = np.eye(4)
        output[:3,:3] = Quaternion(calibrated_data['rotation']).rotation_matrix
        output[:3,3] = calibrated_data['translation']
        
        return output  
    
    def depth_gray(self,depth):
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        # 统计当前深度的相对的数值
        depth = 255 * depth 
        return depth     


traindata = nuScenesDataSet()
# 把 dataset 放入 DataLoader
loader = torch.utils.data.DataLoader(
    dataset=traindata,          # torch TensorDataset format
    batch_size=1,               # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)


"""类别转成独热编码"""

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
for epoch in range(1):   # 训练所有!整套!数据 3 次
    for step, batchdata in enumerate(loader):
        '''搞清楚'''
        gt_img = batchdata['img_labels']
        gt_ego = batchdata['ego_lable']
        gt_seg = batchdata['segmentation']
        cannyTN = batchdata['image_canny']
        images = batchdata['image']
        ego_to_img = batchdata['ego_to_img']
        depths = batchdata['depths']
        print("时序的数量",len(gt_img))
        for t in range(len(gt_img)):
            cams_gt = gt_img[t]
            ego_gt = gt_ego[t]
            bev_gt = gt_seg[t]
            for view,category in enumerate(cams_gt): # 遍历某一个视角下的物体类别和 坐标
                # print(cams[view],"视角的标注物体数量",len(category))
                for anni in category:
                    print(anni[1]) 
                # print(cannyimg.size()) # torch.Size([1, 900, 1600]) 
                
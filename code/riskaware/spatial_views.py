"""Native-event spatial views; never resize an already downsampled event tensor."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class View:
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 480
    size: int = 256

    @property
    def geometry(self):
        ratio=min(self.size/self.width,self.size/self.height)
        w,h=round(self.width*ratio),round(self.height*ratio)
        left,top=round((self.size-w)/2-.1),round((self.size-h)/2-.1)
        return ratio,w,h,left,top

    def encode(self,frames):
        _,w,h,left,top=self.geometry
        result=[]
        for f in frames:
            crop=f[self.y:self.y+self.height,self.x:self.x+self.width]
            resized=cv2.resize(crop,(w,h),interpolation=cv2.INTER_NEAREST)
            result.append(np.pad(resized,((top,self.size-h-top),(left,self.size-w-left))))
        return np.stack(result).reshape(1,3,3,self.size,self.size).astype(np.float32)

    def restore(self,normalized_boxes,clip_crop=False):
        ratio,_,_,left,top=self.geometry
        # Match legacy native conversion: float32 multiplication, float64 offsets.
        boxes=(normalized_boxes*self.size-np.array([left,top,left,top]))/ratio
        boxes+=np.array([self.x,self.y,self.x,self.y])
        if clip_crop:
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],self.x,self.x+self.width)
            boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],self.y,self.y+self.height)
        return boxes


def native_signed_frames(events,start,bin_us=20000):
    x,y=events['x'].astype(np.int64),events['y'].astype(np.int64)
    p=events['p'].astype(np.int16)
    bins=(events['t'].astype(np.int64)-start-1)//bin_us
    if len(x) and (x.min()<0 or x.max()>=640 or y.min()<0 or y.max()>=480 or bins.min()<0 or bins.max()>=9):
        raise ValueError('Unexpected spatial/temporal support')
    if not np.isin(p,[0,1]).all(): raise ValueError('Expected 0/1 polarities')
    frames=np.zeros((9,480,640),dtype=np.int32)
    np.add.at(frames,(bins,y,x),2*p-1)
    return np.sign(frames).astype(np.float32)

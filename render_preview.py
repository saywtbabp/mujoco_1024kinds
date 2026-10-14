"""Optional actual MuJoCo render; PyCharm Run. PNG writer uses the standard library."""
import struct
import zlib
from navlab import runtime
import mujoco
from navlab.config import ROOT,load_config
from navlab.map import MapSpec
from navlab.physics import World,scene_xml


def png(path,array):
    height,width = array.shape[:2]
    def chunk(tag,data):
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    raw = b"".join(b"\0"+array[y].tobytes() for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+
                     chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+
                     chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))


def main():
    cfg,spec = load_config(),MapSpec()
    world = World(spec,cfg)
    world.model.vis.global_.offwidth = 1440
    world.model.vis.global_.offheight = 600
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.07,1.69,0.]
    camera.distance,camera.azimuth,camera.elevation = 7.,90.,-90.
    world.model.vis.global_.fovy = 35.
    option = mujoco.MjvOption(); option.sitegroup[:] = 0
    with mujoco.Renderer(world.model,height=600,width=1440) as renderer:
        renderer.update_scene(world.data,camera=camera,scene_option=option)
        png(ROOT/"scene_preview.png",renderer.render())
    # Exported XML includes the actual original drone by absolute path for inspection.
    (ROOT/"scene.xml").write_text(scene_xml(spec,cfg),encoding="utf-8")
    print("Saved:",ROOT/"scene_preview.png")


if __name__ == "__main__":
    main()

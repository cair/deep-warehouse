import asyncio
from multiprocessing import Process

from aiohttp import web
import cv2
import numpy as np
from aiohttp.web_runner import AppRunner, TCPSite
import SharedArray as sa
import uvloop

class HTTPRenderer(Process):

    def __init__(self, fps=30):
        super().__init__()
        self.data = None
        self.encoding = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        self.fps = fps
        self.fps_wait = 1 / self.fps
        self._loop = None

    def run(self):
        self._loop = uvloop.new_event_loop()
        self._loop.create_task(self.run_server())
        #self._loop.create_task(self.get_data_pointer())
        self._loop.run_forever()

    async def get_data_pointer(self):
        while self.data is None:
            try:
                self.data = sa.attach("shm://env_state")  # FileNotFoundError TODO
            except FileNotFoundError as e:
                pass
            await asyncio.sleep(1)

    async def run_server(self):
        app = web.Application()
        app.add_routes([web.get('/', self.serve_image)])
        runner = AppRunner(app)
        await runner.setup()
        site = TCPSite(runner, "0.0.0.0", 8080)
        await site.start()

    async def serve_image(self, request):
        boundary = "boundarydonotcross"

        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'multipart/x-mixed-replace; '
                            'boundary=--%s' % boundary,
        })

        await response.prepare(request)

        while True:

            if self.data is None:
                await self.get_data_pointer()
            data = await self.generate_state()

            await response.write(
                '--{}\r\n'.format(boundary).encode('utf-8'))
            await response.write(b'Content-Type: image/jpeg\r\n')
            await response.write('Content-Length: {}\r\n'.format(
                len(data)).encode('utf-8'))
            await response.write(b"\r\n")
            # Write data
            await response.write(data)
            await asyncio.sleep(self.fps_wait)

        wc.shutdown()
        return response

    async def generate_state(self):
        state = self.data
        if state is None:
            return None

        result, encimg = cv2.imencode('.jpg', state, self.encoding)
        data = encimg.tostring()
        return data






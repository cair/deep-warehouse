import asyncio
from aiohttp import web
import cv2
import numpy as np
from aiohttp.web_runner import AppRunner, TCPSite


class HTTPRenderer:

    def __init__(self, fps=30, loop=None):
        self.encoding = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        self.fps = fps
        self.fps_wait = 1 / self.fps
        self._loop = loop if loop else asyncio.get_event_loop()
        self.data = np.zeros(shape=(800, 800, 3))

        loop.create_task(self.run_server())

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

    async def blit(self, environment_state):
        self.data = environment_state

    async def generate_state(self):
        result, encimg = cv2.imencode('.jpg', self.data, self.encoding)
        data = encimg.tostring()
        return data






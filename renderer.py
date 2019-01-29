import asyncio
from aiohttp import web
import cv2
import numpy as np
from aiohttp.web_runner import AppRunner, TCPSite


class HTTPRenderer:

    def __init__(self, fps=30, loop=None):
        self.shared_state = None
        self.shared_dimensions = None
        self.encoding = (int(cv2.IMWRITE_JPEG_QUALITY), 90)
        self.fps = fps
        self.fps_wait = 1 / self.fps
        self._loop = loop if loop else asyncio.get_event_loop()

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

    async def generate_state(self):
        self.shared_state = np.frombuffer(self._shared_data.get_obj()).reshape(self._shared_dimensions.get_obj())
        result, encimg = cv2.imencode('.jpg', self.shared_state, self.encoding)
        data = encimg.tostring()
        return data

    def set_shared_state(self, shared_data, shared_dimensions):
        self._shared_data = shared_data
        self._shared_dimensions = shared_dimensions





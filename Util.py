class Util:

    def getDefaultFaceBox(self, frame_width, frame_height):
        frame_left_x = int(frame_width / 2 - 20)
        frame_right_x = int(frame_width / 2 + 20)
        frame_bottom_y = int(frame_height / 2 - 20)
        frame_top_y = int(frame_height / 2 + 20)

        return (frame_left_x, frame_right_x, frame_bottom_y, frame_top_y)

    def getDefaultNoiseBox(self, frame_width, frame_height):
        noise_left_x = 1
        noise_right_x = 20
        noise_bottom_y = int(frame_height - 20)
        noise_top_y = int(frame_height - 1)

        return (noise_left_x, noise_right_x, noise_bottom_y, noise_top_y)
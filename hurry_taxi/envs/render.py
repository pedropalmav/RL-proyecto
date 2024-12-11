import pygame
class Render():
    road_folder = "hurry_taxi/assets/roads/"
    road_sprite = {
        'horizontal': pygame.image.load(road_folder+'horizontal_road.png').convert_alpha(),
        'vertical': pygame.image.load(road_folder+'vertical_road.png').convert_alpha(),
    }
    @staticmethod
    def get_sprite(self, connections):
        road_type = self.get_road_type(connections)
        return self.road_sprite[road_type]
    @staticmethod
    def get_road_type(self, connections):
        connection_count = sum(connections)
        if connection_count == 2:
            if connections["up"] and connections["right"]:
                return "curve_up_right"
            elif connections["up"] and connections["left"]:
                return "curve_up_left"
            elif connections["down"] and connections["right"]:
                return "curve_down_right"
            elif connections["down"] and connections["left"]:
                return "curve_down_left"
            elif connections["up"] and connections["down"]:
                return "vertical"
            elif connections["left"] and connections["right"]:
                return "horizontal"
        elif connection_count == 4:
            return "crossroad"  # Connected to all 4 sides
        elif connection_count == 3:
            if not connections["up"]:
                return "T_down"
            elif not connections["down"]:
                return "T_up"
            elif not connections["left"]:
                return "T_right"
            elif not connections["right"]:
                return "T_left"
        elif connection_count == 1:
            if connections["up"]:
                return "end_down"
            elif connections["down"]:
                return "end_up"
            elif connections["left"]:
                return "end_right"
            elif connections["right"]:
                return "end_left"
        return "building"
    @staticmethod
    def get_connections(self, map, x, y):
        connections = {
            "up": map[y - 1][x],
            "down": map[y + 1][x],
            "left": map[y][x - 1],
            "right": map[y][x + 1],
        }
        
        return connections


            

        

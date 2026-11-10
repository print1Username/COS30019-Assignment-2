import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Matplotlib:
    def __init__(self, width, height, title, x_label, y_label, color="blue", grid=True, alpha=1):
        self.title = title
        self.width = width  # The width for the diagram
        self.height = height  # The height for the diagram
        self.x_label = x_label
        self.y_label = y_label
        self.color = color  # The color for the chart (Default Blue)
        self.grid = grid  # Generate the grid in the diagram (Default: True)
        self.alpha = alpha  # The shade for the grid line (Default: 1)

        self.dictionary = {}
        self.x_list = []
        self.y_list = []

    def AddData(self, x_value, y_value):
        self.x_list.append(x_value)
        self.y_list.append(y_value)

    def AddDataList(self, x_value_list, y_value_list):
        self.x_list += x_value_list
        self.y_list += y_value_list

    def ClearData(self):
        self.x_list = []
        self.y_list = []
        self.dictionary = {}

    def AddDictionary(self, x_value, y_value):
        self.dictionary[x_value] = y_value

    def AddDictionaryList(self, dictionary):
        self.dictionary = dictionary.copy()

    '''
    Convert the dictionary to a list    
    dictionary = {
        number 1: 11,
        number 2: 22,
        number 3: 33
    }
    
    After Convert:
    x list = [number1, number2, number3]
    y list = [11, 22, 33]
    '''

    def ConvertDictionaryToList(self):
        self.x_list = []
        self.y_list = []

        for dict in self.dictionary:
            self.x_list.append(dict)
            self.y_list.append(self.dictionary[dict])

    # Create the chart and save it
    # If window is True, then it will generate the window, else it will not generate the windows.
    def CreateChart(self, diagram_type, window=True, save_format="jpg", save_directory=None):
        try:
            plt.figure(figsize=(self.width, self.height))
            plt.title(self.title)
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            plt.grid(self.grid, alpha=self.alpha)

            if window is not True:
                plt.ioff()

            match diagram_type:
                case 1:  # Line Chart 线状图
                    plt.plot(self.x_list, self.y_list, color=self.color)
                case 2:  # Scatter Chart 散点图
                    plt.scatter(self.x_list, self.y_list, color=self.color, s=50)
                case 3:  # Bar Chart 柱状图
                    plt.bar(self.x_list, self.y_list, color=self.color)
                case 4:  # Pie Chart 饼图
                    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
                    plt.pie(self.y_list, labels=self.x_list, colors=colors, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                case _:
                    print("ERROR: Invalid chart type.")

            save_format_list = ["jpg", "jpeg", "png", "pdf"]

            if save_format.lower() not in save_format_list:
                print(f"⚠️ ERROR: Can't save the file with '{save_format}'. Supported formats: {save_format_list}")
                return False

            try:
                # If the save directory have value, it will save in the directory
                # else it will save as the directory same as the file
                if save_directory:
                    os.makedirs(save_directory, exist_ok=True)
                    file_path = os.path.join(save_directory, f"{self.title}.{save_format.lower()}")
                else:
                    file_path = f"{self.title}.{save_format.lower()}"

                plt.savefig(file_path, bbox_inches='tight', dpi=300)
                plt.close()

                print(f"Chart saved successfully: {file_path}")
                return True

            # Output the error if the chart cannot save
            except Exception as e:
                print(f"⚠️ ERROR: Failed to save chart: {str(e)}")
                return False

        except Exception as e:
            print(f"⚠️ ERROR for the chart: {str(e)}")


if __name__ == "__main__":
    print()

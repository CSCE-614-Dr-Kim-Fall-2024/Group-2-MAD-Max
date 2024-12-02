import matplotlib.pyplot as plt
import os
import json
import logging

logger = logging.getLogger()
# logger.setLevel(logging.INFO)

class Node():
    def __init__(self):
        self.parent = []
        self.child = []
        self.ref = 0
        self.note = None
        
    def add_child(self, child):
        self.child.append(child)
    
    def add_parent(self, parent):
        self.parent.append(parent)
    
    def is_child(self, node):
        # node1.is_chlid(node2) : True  if node2 is child of node1,
        #                         False otherwise
        return node in self.parent
    
    def is_parent(self, node):
        # node1.is_parent(node2) : True  if node2 is parent of node1,
        #                          False otherwise
        return node in self.child

    def add_dependency(self, to):
        # check cyclic dependency
        if self.is_child(to):
            logger.error(f"add_dependency - {self} is a child of {to}")
            return
        
        if to.is_child(self) and self.is_parent(to):
            return

        if not to.is_child(self):
            to.add_parent(self)
        if not self.is_parent(to):
            self.add_child(to)

        to.ref = len(to.parent)

    def del_dependency(self, child):
        if child not in self.child:
            logger.error(f"del_dependency - {child} is not a child of {self}")
        
        self.child.remove(child)
        child.parent.remove(self)

        child.ref = len(child.parent)
    
    def is_comm_node(self):
        return False


class LayerNode(Node):
    '''
        temporary node of dependency graph
            which contains framework-level dependencies
        i.e. inter-layer dependencies without low-level information 
            (such as GPU kernel executions)
    '''
    def __init__(self, layer_num, layer_name, function, stream):
        super(LayerNode, self).__init__()

        self.layer_num = layer_num
        self.layer_name = layer_name
        self.stream = stream
        self.duration = 0
        self.start = 0
        self.gap = 0
        
        self.function = function
    
    def __repr__(self) -> str:
        return self.function


class CommNode(Node):
    '''
        temporary node of dependency graph
            which contains framework-level dependencies
        i.e. inter-layer dependencies without low-level information 
            (such as GPU kernel executions)
    '''
    def __init__(self, bucket_size, stream="Comm", function="allreduce"):
        super(CommNode, self).__init__()
        self.stream = stream
        self.bucket_size = bucket_size
        self.duration = 0
        self.start = 0
        self.gap = 0

        self.function = function
    
    def __repr__(self) -> str:
        return f"{self.function} (size={self.bucket_size/1024/1024:.2f}MB)"

    def is_comm_node(self):
        return True


class TaskNode(Node):
    def __init__(self, duration, name, stream, cid, gap):
        super(TaskNode, self).__init__()
        self.duration = duration
        self.name = name
        self.stream = stream
        self.cid = cid
        self.start = 0
        self.gap = gap

        self.function = ""

    def __repr__(self) -> str:
        return f"{self.name}"




class DepGraph():
    '''
        doc-string
    '''

    def __init__(self):
        self.streams = dict()

    
    def create_stream(self, stream):
        if stream not in self.streams.keys():
            self.streams[stream] = []


    def add_node(self, node, prev=[]):
        # if prev is None:
        #     print ("[ERROR] add_node: one of ``stream`` and ``prev`` must be specified")
        #     return False

        stream = node.stream
        # if self.streams[stream]:
        #     self.add_dependency(self.streams[stream][-1], node)
        self.streams[stream].append(node)

        for prevNode in prev:
            self.add_dependency(prevNode, node)
    
    def append_node_to_stream(self, node, stream):
        self.streams[stream].append(node)
    
    def add_dependency(self, parent, child):
        parent.add_dependency(child)


    def del_dependency(self, parent, child):
        parent.del_dependency(child)


    '''
        helper functions for debugging
    '''
    def find_overlap(timeline):
        overlap_duration = 0
        for stream1, data1 in timeline.items():
            for stream2, data2 in timeline.items():
                if stream1 != stream2:
                    for comm_start, comm_duration in data1["comm"]:
                        for comp_start, comp_duration in data2["comp"]:
                            overlap_start = max(comm_start, comp_start)
                            overlap_end = min(comm_start + comm_duration, comp_start + comp_duration)
                            if overlap_end > overlap_start:
                                overlap_duration += overlap_end - overlap_start
        txt_file_path = os.path.join(os.getcwd(), "overlap_duration.txt")
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(f"Total overlap duration: {total_overlap} units")
        
        print(f"Overlap duration written to {txt_file_path}")
    # return overlap_duration



    def print_graph(self):
        for stream in self.streams.keys():
            if len(self.streams[stream]) == 0:
                continue
            print (f"[{stream}]")

            print (self.streams[stream][0], end=" ")
            nodeCnt = 1
            for node in self.streams[stream][1:]:
                print (f"-> {node}", end=" ")
                nodeCnt += 1
                if nodeCnt >= 8:
                    print ("")
                    nodeCnt = 0
            print ("\n")
        
        print ("[Inter-stream dependencies]")
        for stream in self.streams.keys():
            if len(self.streams[stream]) == 0:
                continue
            
            for node in self.streams[stream]:
                cand = [c for c in node.child if c.stream != stream]
                for c in cand:
                    print (f"{node} [{node.stream}] -> {c} [{c.stream}]")

        # for node in self.streams["Comm"]:
        #     for p in node.parent:
        #         print (f"{p} ({p.stream}) -> {node} ({node.stream})")

        print ("")

    

    def show_graph(self, plot_file, overlap_file):
        timeline = {}
        for stream, tasks in self.streams.items():
            if stream == "Comm":
                continue
            if len(tasks) == 0:
                continue
            # timeline[stream] = []
            timeline[stream] = {"fwd": [], "bwd": [], "wu": [], "comm": []}
            for u in tasks:
                stage = u.function.split("_")[0].lower()
                if stage == "allreduce":
                    stage = "comm"
                timeline[stream][stage].append((u.start, u.duration))


        plt.close('all')
        px = 1/plt.rcParams['figure.dpi']

        num_stream = len(timeline)
        fig = plt.figure(figsize=(1600*px, 60*num_stream*px))
        ax = fig.subplots()
        for i, stream in enumerate(reversed(timeline.keys())):
            for stage, nodes in timeline[stream].items():
                if stage == 'fwd':
                    color = ('powderblue', 'lightskyblue', 'dodgerblue')
                elif stage == 'bwd':
                    color = ('greenyellow', 'limegreen', 'forestgreen')
                elif stage == 'wu':
                    color = ('lightcoral', 'tab:red')
                else:
                    color = ('darkgrey')
                # print (len(timeline[stream]))
                ax.broken_barh(nodes, (1+i*5, 4), facecolors=color)

        plt.axis('tight')
        ax.set_ylim((0, 1+5*num_stream))
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        # variant_name = filename[filename.rindex('/')+1:]
        # plt.savefig("%s/%s_plot.png"%(filename,variant_name))
        plt.savefig(plot_file)

        # # Save the graph as an image file in the same folder
        # file_path = os.path.join(os.getcwd(), "graph_output.png")
        # plt.savefig(file_path, bbox_inches='tight')
        # print(f"Graph saved to {file_path}")
        # cwd = os.getcwd()
        # plt.show()
        # plt.savefig("%s/results/%s_streamplot.png"%(cwd,filename))

        
        # Save timeline to JSON file with simplified structure
        # json_file_path = os.path.join(os.getcwd(), "timeline_data.json")
        # with open(json_file_path, 'w') as json_file:
            # Convert tuples to lists for JSON serialization
        serializable_timeline = {}
        for stream, stages in timeline.items():
            serializable_timeline[stream] = {
                "comp": [],
                "comm": timeline[stream]["comm"]
            }
            # Combine fwd, bwd, wu into comp for JSON
            for stage in ["fwd", "bwd", "wu"]:
                serializable_timeline[stream]["comp"].extend(
                    [list(event) for event in timeline[stream][stage]]
                )
        
            # json.dump(serializable_timeline, json_file, indent=4)

        overlap_duration = 0
        streams = list(serializable_timeline.keys())
    
        for i in range(len(streams) - 1):
            current_stream = streams[i]
            next_stream = streams[i + 1]
            
            # Compare current stream's comm with next stream's comp
            for comm_start, comm_duration in serializable_timeline[current_stream]["comm"]:
                for comp_start, comp_duration in serializable_timeline[next_stream]["comp"]:
                    overlap_start = max(comm_start, comp_start)
                    overlap_end = min(comm_start + comm_duration, comp_start + comp_duration)
                    
                    if overlap_end > overlap_start:
                        overlap_duration += overlap_end - overlap_start
                       
        
        # for stream1, data1 in serializable_timeline.items():
        #     for stream2, data2 in serializable_timeline.items():
        #         if stream1 != stream2:
        #             logger.info(f"{stream1}, {stream2}")
        #             for comm_start, comm_duration in data1["comm"]:
        #                 for comp_start, comp_duration in data2["comp"]:
        #                     overlap_start = max(comm_start, comp_start)
        #                     overlap_end = min(comm_start + comm_duration, comp_start + comp_duration)
        #                     if overlap_end > overlap_start:
        #                         overlap_duration += overlap_end - overlap_start
        
        # Calculate total communication time
        total_comm_time = 0
        for stream, data in serializable_timeline.items():
            logger.info(f"{stream}'s overlap")
            if data["comm"]:
                num_comm_events = len(data["comm"])
                comm_duration = data["comm"][0][1]  # duration of first comm event
                stream_comm_time = num_comm_events * comm_duration
                total_comm_time += stream_comm_time
        comm_time_idle = total_comm_time - overlap_duration
        overlap_percentage = overlap_duration/total_comm_time 



        # txt_file_path = "%s/%s_overlap.txt"%(overlafile)
        # txt_file_path = "%s/%s_overlap.txt"%(filename,variant_name)
        # txt_file_path = os.path.join(os.getcwd(), "/results/%s_overlap.txt"%(filename))
        with open(overlap_file, 'w') as txt_file:
            txt_file.write(f"Total overlap duration: {overlap_duration} units\n")
            txt_file.write(f"Total comm duration: {total_comm_time} units\n")
            txt_file.write(f"Total comm Idle duration: {comm_time_idle} units\n")
            txt_file.write(f"Overlap Percentage: {overlap_percentage} units\n")
        
        print(f"Overlap duration written to {overlap_file}")
     
    
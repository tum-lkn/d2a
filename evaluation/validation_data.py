import json
from environment import problem_generator_2 as pgen
from typing import List, Dict

PATH_TO_DATAFILE = '/opt/project/EvalData/ExampleProblems_200.json'

class ProblemData(object):
    @classmethod
    def from_dict(cls, dict: Dict):
        return ProblemData(**dict)       

    def __init__(self, sfcs: List[pgen.Sfc] = None, tx_count: int = None) -> None:
        self.sfcs = sfcs
        self.tx_count = tx_count

    def to_dict(self):
        return {
            'tx_count': self.tx_count,
            'sfcs': [sfc.to_dict() for sfc in self.sfcs]
        } 
    

class ProblemCollection(object):
    @classmethod
    def from_dict(cls, dict: Dict):
        return ProblemCollection(**dict)

    def __init__(self, problem_list: List[ProblemData] = None) -> None:
        self.problem_count: int = 0
        if problem_list != None:
            self.problem_count = len(problem_list)
        self.problem_list: List[ProblemData] = problem_list

    def to_dict(self):
        ret_dict = {}
        ret_dict['problem_count'] = self.problem_count
        ret_dict['problem_list'] = [problem.to_dict() for problem in self.problem_list]
        return ret_dict
        

def generate_data(samples: int):
    problems = []

    all_machines = [pgen.Machine(i, 2.2e9) for i in range(3, 22)]
    problem_generator = pgen.HeavyTailedGeneratorWithTx(
        machines=all_machines,
        num_sfcs=8,
        max_num_vnfs_per_sfc=8,
        load_level=0.9,
        max_num_vnfs_in_system=39,
        system_rate=5e6,
        seed=5071998,
        num_tx=5
    )

    for i in range(samples):
        sfcs, pl, tx = problem_generator.__next__()
        problems.append(ProblemData(sfcs, len(tx)))


    collection = ProblemCollection(problems)

    with open(PATH_TO_DATAFILE, 'w+') as file:
        json.dump(collection.to_dict(), file, indent=4)
        file.close()

def load_problems(path) -> ProblemCollection:
    file = open(path, 'r')

    if not file:
        raise 'File no found'

    json_data = json.load(file)
    prob_colletion = ProblemCollection()
    prob_colletion.problem_count = json_data['problem_count']
    prob_colletion.problem_list = []

    for problem in json_data['problem_list']:
        cur_p = ProblemData()
        cur_p.tx_count = problem['tx_count']
        cur_p.sfcs = []
        for sfc in problem['sfcs']:
            jobs = []
            for job in sfc['jobs']:
                vnf = pgen.Vnf(job['vnf']['compute_per_packet'])
                vnf.instance_id = job['vnf']['instance_id']
                cur_job = pgen.Job(job['rate'], vnf)
                jobs.append(cur_job)

            cur_sfc = pgen.Sfc(jobs, sfc['rate'], sfc['weight'])
            cur_p.sfcs.append(cur_sfc)

        prob_colletion.problem_list.append(cur_p)

    return prob_colletion


if __name__ == '__main__':
    generate_data(200)
    #load_problems()
#pragma once
#include "BasicGraph.h"
#include "States.h"
#include "PriorityGraph.h"
#include "PBS.h"
#include "WHCAStar.h"
#include "ECBS.h"
#include "LRAStar.h"
#include <cstdio>
#include <sys/types.h>


class BasicSystem
{
public:
    // params for MAPF algotithms
	MAPFSolver& solver;
	bool hold_endpoints;
	bool useDummyPaths;
    int time_limit;
    int travel_time_window;
	//string potential_function;
	//double potential_threshold;
	//double suboptimal_bound;
    int screen;
	bool log;
    int num_of_drives;
    int seed;
    int simulation_window;
    int planning_window;
    int simulation_time;
    bool use_learned_cost = false;
    std::string learned_cost_ckpt;
    std::string learned_cost_python = "python3";
    int learned_cost_history_len = 5;
    double learned_cost_weight = 1.0;
    double gaussian_sigma = 1.5;
    double pred_bias = 0.0;
    int gaussian_ksize = 9;
    bool learned_cost_normalize = false;

    // params for drive model
    bool consider_rotation;
    int k_robust;

    BasicSystem(const BasicGraph& G, MAPFSolver& solver);
    ~BasicSystem();

	// TODO
    /*bool load_config(std::string fname);
    bool generate_random_MAPF_instance();
    bool run();
	void print_MAPF_instance() const;
	void save_MAPF_instance(std::string fname) const;
	bool read_MAPF_instance(std::string fname);*/

    // I/O
    std::string outfile;
    void save_results();
	double saving_time = 0; // time for saving results to files, in seconds
    int num_of_tasks; // number of finished tasks

	list<int> new_agents; // used for replanning a subgroup of agents

    // used for MAPF instance
    vector<State> starts;
    vector< vector<pair<int, int> > > goal_locations;
	// unordered_set<int> held_endpoints;
    int timestep;

    // record movements of drives
    std::vector<Path> paths;
    std::vector<std::list<std::pair<int, int> > > finished_tasks; // location + finish time

    bool congested() const;
	bool check_collisions(const vector<Path>& input_paths) const;

    // update
    void update_start_locations();
    void update_travel_times(unordered_map<int, double>& travel_times);
    bool update_learned_costs();
    void update_paths(const std::vector<Path*>& MAPF_paths, int max_timestep);
    void update_paths(const std::vector<Path>& MAPF_paths, int max_timestep);
    void update_initial_paths(vector<Path>& initial_paths) const;
    void update_initial_constraints(list< tuple<int, int, int> >& initial_constraints) const;
    
	void add_partial_priorities(const vector<Path>& initial_paths, PriorityGraph& initial_priorities) const;
	list<tuple<int, int, int>> move(); // return finished tasks
	void solve();
	void initialize_solvers();
	bool load_records();
	bool load_locations();


protected:
	bool solve_by_WHCA(vector<Path>& planned_paths,
		const vector<State>& new_starts, const vector< vector<pair<int, int> > >& new_goal_locations);
    bool LRA_called = false;

private:
	const BasicGraph& G;
    bool run_learned_cost_inference(const std::string& input_file, const std::string& output_file) const;
    bool start_learned_cost_worker();
    void stop_learned_cost_worker();
    bool infer_learned_cost_via_worker(
        const std::vector<int>& obstacles,
        const std::vector<float>& occupancy,
        const std::vector<float>& vertical,
        const std::vector<float>& horizontal,
        int rows,
        int cols,
        int history_len,
        std::vector<double>& costs);
    pid_t learned_cost_worker_pid = -1;
    FILE* learned_cost_worker_in = nullptr;
    FILE* learned_cost_worker_out = nullptr;
};

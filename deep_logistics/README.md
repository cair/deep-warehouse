# logistics-sim
A logistics environment where the goal is to use agents to transfer goods between destinations. The agent must baster long-term planning, short-term planning, and clever exploration to master the environment in a sufficient amount of time.

## Installation
```python
pip install git+https://github.com/perara/deep-logistics.git
```
## Run using docker
To increase sample throughput, the docker containers use a shared volume
to store experience replay packs. These packs can then be loaded as training 
samples for the rl-algorithm (Typically inside another container with access
to the same volume)
```
docker volume create --name deep-logistics
```
To run a single agent, run the following command:
```
docker run -d --volume deep-logistics perara/deep-logistics
```

## Environment Specifications

### Constrains and Assumptions
* The agent cannot select direction actions unless standing still
* Assumes the grid to have quadratic cells.

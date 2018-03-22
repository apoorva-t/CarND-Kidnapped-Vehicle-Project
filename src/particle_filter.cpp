/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits.h>
#include <random>

#include "particle_filter.h"

using namespace std;

#define FLTMAX numeric_limits<double>::max();

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000;
	//initialize weights to 1
	weights = std::vector<double> (num_particles,1);

	//initialize particle position drawing from Gaussian distibution around GPS meas.
	std::default_random_engine gen;
	std::normal_distribution<double> distX(x, std[0]);
	std::normal_distribution<double> distY(y, std[1]);
	std::normal_distribution<double> distTheta(theta, std[2]);
	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i+1; p.x = distX(gen);
		p.y = distY(gen); p.theta = distTheta(gen);
		p.weight = 1;
		particles.push_back(p);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::normal_distribution<double> distX(0,std_pos[0]);
	std::normal_distribution<double> distY(0,std_pos[1]);
	std::normal_distribution<double> distTheta(0,std_pos[2]);
	std::default_random_engine gen;
	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) < 0.001)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			double factor = velocity/yaw_rate;
			double theta_eff = particles[i].theta + yaw_rate*delta_t;
			particles[i].x += factor * (sin(theta_eff) - sin(particles[i].theta));
			particles[i].y += factor * (cos(particles[i].theta) - cos(theta_eff));
			particles[i].theta += yaw_rate*delta_t;
		}
		particles[i].x += distX(gen);
		particles[i].y += distY(gen);
		particles[i].theta += distTheta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++)
	{
		double minDist = FLTMAX;
		double d = 0.0;
		int id_f = -1;
		for (int j = 0; j < predicted.size(); j++)
		{
			if ( ( d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y)) < minDist)
			{
				minDist = d;
				id_f = predicted[j].id;
			}
		}
		observations[i].id = id_f;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double gauss_norm = 1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	double var_x = 2.0 * (pow(std_landmark[0],2));
	double var_y = 2.0 * (pow(std_landmark[1],2));
	for (int i = 0; i < num_particles; i++)
	{
		//transform observations to MAP co-ordinate system
		std::vector<LandmarkObs> transf_obs = observations;

		double x_p = particles[i].x, y_p = particles[i].y;
		double theta_p = particles[i].theta;
		for (int j = 0; j < transf_obs.size(); j++)
		{
			transf_obs[j].x = observations[j].x*cos(theta_p) - observations[j].y*sin(theta_p) + x_p;
			transf_obs[j].y = observations[j].x*sin(theta_p) + observations[j].y*cos(theta_p) + y_p;
		}

		// find associated landmarks for all observations
		vector<LandmarkObs> m_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];

			// consider landmark only if within sensor range
			if (dist(lm.x_f, lm.y_f, x_p, y_p) > sensor_range) continue;

			LandmarkObs lmOb;
			lmOb.id = lm.id_i;
			lmOb.x = lm.x_f;
			lmOb.y = lm.y_f;

			m_landmarks.push_back(lmOb);
		}
		dataAssociation(m_landmarks, transf_obs);
		//use multi-variate gaussian pdf to update weight of particle
		double prod = 1.0;
		for (int j = 0; j < transf_obs.size(); j++)
		{
			LandmarkObs obs = transf_obs[j];
			//find landmark associated with this observation
			double map_x, map_y;
			for (int k = 0; k < m_landmarks.size(); k++)
			{
				if (m_landmarks[k].id == obs.id)
				{
					map_x = m_landmarks[k].x;
					map_y = m_landmarks[k].y;
					break;
				}
			}
			double expo = (pow((obs.x - map_x),2)/var_x + pow((obs.y - map_y),2)/var_y);
			prod *= gauss_norm * exp(-expo);
		}
		particles[i].weight = prod;
		weights[i] = prod;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine gen;
	std::discrete_distribution<double> dist(weights.begin(), weights.end());
	vector<Particle> resampledParticles;
	for (int i = 0; i < num_particles; i++)
	{
		int p = dist(gen);
		resampledParticles.push_back(particles[p]);
		//cout << "particle num: " << p << ", weight: " << weights[p] << std::endl;
	}
	particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

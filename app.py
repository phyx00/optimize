import streamlit as st
import pandas as pd
import numpy as np
import pulp
import altair as alt


    


# Parameter ranges
parameter_ranges = {
    'budget': {'min': 100000, 'max': 1000000, 'step': 100000},
    'romi_target': {'min': 0.0, 'max': 2.0, 'step': 0.01},
    'cpa_test': {'min': 10, 'max': 100, 'step': 5},
    'cpa_boost': {'min': 10, 'max': 100, 'step': 5},
    'cpa_scale': {'min': 10, 'max': 100, 'step': 5},
    'cpa_alpha': {'min': 10, 'max': 100, 'step': 5},
    'conv_test': {'min': 1, 'max': 50, 'step': 1},
    'conv_boost': {'min': 1, 'max': 100, 'step': 5},
    'conv_scale': {'min': 10, 'max': 1000, 'step': 10},
    'conv_alpha': {'min': 100, 'max': 1000, 'step': 10},
    'rev_conv_test': {'min': 10, 'max': 100, 'step': 5},
    'rev_conv_boost': {'min': 10, 'max': 100, 'step': 5},
    'rev_conv_scale': {'min': 10, 'max': 100, 'step': 5},
    'rev_conv_alpha': {'min': 10, 'max': 100, 'step': 5},
    'alpha_rate': {'min': 0.1, 'max': 1.0, 'step': 0.05},
    'beta_rate': {'min': 0.1, 'max': 1.0, 'step': 0.05},
    'gamma_rate': {'min': 0.1, 'max': 1.0, 'step': 0.05}
}

from sklearn.ensemble import RandomForestRegressor

def generate_optimization_dataset(num_samples=100):
    data = []
    for _ in range(num_samples):
        sample_params = {}
        for param, info in parameter_ranges.items():
            if isinstance(info['min'], int):
                sample_params[param] = np.random.randint(info['min'], info['max'] + 1)
            else:
                sample_params[param] = np.random.uniform(info['min'], info['max'])
        
        result = optimize_creatives_pulp_cached(**sample_params)
        
        if result['status'] == 'Optimal':
            sample_params['ROMI'] = result['actual_romi']
            data.append(sample_params)
        else:
            # Optionally handle non-optimal cases or skip them
            pass
    return pd.DataFrame(data)

def optimize_creatives_pulp(
    budget,
    romi_target,
    cpa_test,
    cpa_boost,
    cpa_scale,
    cpa_alpha,
    conv_test,
    conv_boost,
    conv_scale,
    conv_alpha,
    rev_conv_test,
    rev_conv_boost,
    rev_conv_scale,
    rev_conv_alpha,
    alpha_rate,
    beta_rate,
    gamma_rate
):
    # Define the problem
    prob = pulp.LpProblem("Creative_Optimization", pulp.LpMaximize)
    
    # Decision variables (integers)
    n_test = pulp.LpVariable('n_test', lowBound=0, cat='Integer')
    n_boost = pulp.LpVariable('n_boost', lowBound=0, cat='Integer')
    n_scale = pulp.LpVariable('n_scale', lowBound=0, cat='Integer')
    n_alpha = pulp.LpVariable('n_alpha', lowBound=2, upBound=5, cat='Integer')
    
    # Objective function: Maximize Total Revenue
    total_revenue = (
        rev_conv_test * conv_test * n_test +
        rev_conv_boost * conv_boost * n_boost +
        rev_conv_scale * conv_scale * n_scale +
        rev_conv_alpha * conv_alpha * n_alpha
    )
    
    prob += total_revenue, "Total Revenue"
    
    # Total Cost
    total_cost = (
        cpa_test * conv_test * n_test +
        cpa_boost * conv_boost * n_boost +
        cpa_scale * conv_scale * n_scale +
        cpa_alpha * conv_alpha * n_alpha
    )
    
    # Budget Constraint
    prob += total_cost <= budget, "Budget Constraint"
    
    # ROMI Constraint
    prob += total_revenue - (1 + romi_target) * total_cost >= 0, "ROMI Constraint"
    
    # Transition Constraints
    prob += n_boost - alpha_rate * n_test <= 0, "Transition Constraint 1"
    prob += n_scale - beta_rate * n_boost <= 0, "Transition Constraint 2"
    prob += n_alpha - gamma_rate * n_scale <= 0, "Transition Constraint 3"
    
    # Solve the problem
    prob.solve()
    
    # Check if the optimization was successful
    if pulp.LpStatus[prob.status] == 'Optimal':
        # Retrieve the optimized number of creatives
        n_test_value = n_test.varValue
        n_boost_value = n_boost.varValue
        n_scale_value = n_scale.varValue
        n_alpha_value = n_alpha.varValue
        
        # Calculate total cost and revenue
        total_cost_value = pulp.value(total_cost)
        total_revenue_value = pulp.value(total_revenue)
        
        # Calculate actual ROMI
        if total_cost_value > 0:
            actual_romi = (total_revenue_value - total_cost_value) / total_cost_value
        else:
            actual_romi = float('inf')
        
        # Prepare data for charts
        stages = ['Test', 'Boost', 'Scale', 'Alpha']
        creatives = [n_test_value, n_boost_value, n_scale_value, n_alpha_value]
        cost_per_stage = [
            cpa_test * conv_test * n_test_value,
            cpa_boost * conv_boost * n_boost_value,
            cpa_scale * conv_scale * n_scale_value,
            cpa_alpha * conv_alpha * n_alpha_value
        ]
        revenue_per_stage = [
            rev_conv_test * conv_test * n_test_value,
            rev_conv_boost * conv_boost * n_boost_value,
            rev_conv_scale * conv_scale * n_scale_value,
            rev_conv_alpha * conv_alpha * n_alpha_value
        ]
        profit_per_stage = [
            revenue_per_stage[i] - cost_per_stage[i] for i in range(len(stages))
        ]
        
        result = {
            'status': 'Optimal',
            'n_test': n_test_value,
            'n_boost': n_boost_value,
            'n_scale': n_scale_value,
            'n_alpha': n_alpha_value,
            'total_cost': total_cost_value,
            'total_revenue': total_revenue_value,
            'actual_romi': actual_romi,
            'stages': stages,
            'creatives': creatives,
            'cost_per_stage': cost_per_stage,
            'revenue_per_stage': revenue_per_stage,
            'profit_per_stage': profit_per_stage
        }
    else:
        result = {
            'status': pulp.LpStatus[prob.status],
            'message': 'Optimization failed. Please adjust the parameters and try again.'
        }
    
    return result


def main():
    st.title("Interactive Creative Optimization Model")
    
    st.sidebar.header("Input Parameters")
    
    # Budget and ROMI Target
    #budget = st.sidebar.number_input('Budget', value=500000, step=10000)
    budget = st.sidebar.number_input('Budget',
                                 value=600000,
                                 min_value=parameter_ranges['budget']['min'],
                                 max_value=parameter_ranges['budget']['max'],
                                 step=parameter_ranges['budget']['step'])

    #romi_target = st.sidebar.slider('ROMI Target', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    romi_target = st.sidebar.slider('ROMI Target',
                                min_value=parameter_ranges['romi_target']['min'],
                                max_value=parameter_ranges['romi_target']['max'],
                                value=0.3,
                                step=parameter_ranges['romi_target']['step'])
    
    st.sidebar.subheader("Cost Per Acquisition (CPA)")
    #cpa_test = st.sidebar.slider('CPA Test', min_value=10, max_value=100, value=50, step=1)
    cpa_test = st.sidebar.slider('CPA Test',
                             min_value=parameter_ranges['cpa_test']['min'],
                             max_value=parameter_ranges['cpa_test']['max'],
                             value=75,
                             step=parameter_ranges['cpa_test']['step'])
    #cpa_boost = st.sidebar.slider('CPA Boost', min_value=10, max_value=100, value=45, step=1)
    cpa_boost = st.sidebar.slider('CPA Boost',
                             min_value=parameter_ranges['cpa_boost']['min'],
                             max_value=parameter_ranges['cpa_boost']['max'],
                             value=55,
                             step=parameter_ranges['cpa_boost']['step'])
    #cpa_scale = st.sidebar.slider('CPA Scale', min_value=10, max_value=100, value=40, step=1)
    cpa_scale = st.sidebar.slider('CPA Scale',
                             min_value=parameter_ranges['cpa_scale']['min'],
                             max_value=parameter_ranges['cpa_scale']['max'],
                             value=45,
                             step=parameter_ranges['cpa_scale']['step'])
    #cpa_alpha = st.sidebar.slider('CPA Alpha', min_value=10, max_value=100, value=35, step=1)
    cpa_alpha = st.sidebar.slider('CPA Alpha',
                             min_value=parameter_ranges['cpa_alpha']['min'],
                             max_value=parameter_ranges['cpa_alpha']['max'],
                             value=30,
                             step=parameter_ranges['cpa_alpha']['step'])
    
    st.sidebar.subheader("Conversions per Creative")
    #conv_test = st.sidebar.slider('Conv Test', min_value=10, max_value=1000, value=100, step=10)
    conv_test = st.sidebar.slider('Conv Test',
                              min_value=parameter_ranges['conv_test']['min'],
                              max_value=parameter_ranges['conv_test']['max'],
                              value=5,
                              step=parameter_ranges['conv_test']['step'])
    
    #conv_boost = st.sidebar.slider('Conv Boost', min_value=10, max_value=2000, value=150, step=10)
    conv_boost = st.sidebar.slider('Conv Boost',
                              min_value=parameter_ranges['conv_boost']['min'],
                              max_value=parameter_ranges['conv_boost']['max'],
                              value=10,
                              step=parameter_ranges['conv_boost']['step'])
    #conv_scale = st.sidebar.slider('Conv Scale', min_value=10, max_value=3000, value=200, step=10)
    conv_scale = st.sidebar.slider('Conv Scale',
                              min_value=parameter_ranges['conv_scale']['min'],
                              max_value=parameter_ranges['conv_scale']['max'],
                              value=150,
                              step=parameter_ranges['conv_scale']['step'])
    #conv_alpha = st.sidebar.slider('Conv Alpha', min_value=10, max_value=4000, value=300, step=10)
    conv_alpha = st.sidebar.slider('Conv Alpha',
                              min_value=parameter_ranges['conv_alpha']['min'],
                              max_value=parameter_ranges['conv_alpha']['max'],
                              value=500,
                              step=parameter_ranges['conv_alpha']['step'])
    
    st.sidebar.subheader("Revenue per Conversion")
    #rev_conv_test = st.sidebar.slider('Rev/Conv Test', min_value=10, max_value=200, value=60, step=5)
    rev_conv_test = st.sidebar.slider('Rev/Conv Test',
                                  min_value=parameter_ranges['rev_conv_test']['min'],
                                  max_value=parameter_ranges['rev_conv_test']['max'],
                                  value=50,
                                  step=parameter_ranges['rev_conv_test']['step'])
    #rev_conv_boost = st.sidebar.slider('Rev/Conv Boost', min_value=10, max_value=200, value=65, step=5)
    rev_conv_boost = st.sidebar.slider('Rev/Conv Boost',
                                  min_value=parameter_ranges['rev_conv_boost']['min'],
                                  max_value=parameter_ranges['rev_conv_boost']['max'],
                                  value=50,
                                  step=parameter_ranges['rev_conv_boost']['step'])
    #rev_conv_scale = st.sidebar.slider('Rev/Conv Scale', min_value=10, max_value=200, value=70, step=5)
    rev_conv_scale = st.sidebar.slider('Rev/Conv Scale',
                                  min_value=parameter_ranges['rev_conv_scale']['min'],
                                  max_value=parameter_ranges['rev_conv_scale']['max'],
                                  value=50,
                                  step=parameter_ranges['rev_conv_scale']['step'])
    #rev_conv_alpha = st.sidebar.slider('Rev/Conv Alpha', min_value=10, max_value=200, value=80, step=5)
    rev_conv_alpha = st.sidebar.slider('Rev/Conv Alpha',
                                  min_value=parameter_ranges['rev_conv_alpha']['min'],
                                  max_value=parameter_ranges['rev_conv_alpha']['max'],
                                  value=100,
                                  step=parameter_ranges['rev_conv_alpha']['step'])
    
    st.sidebar.subheader("Transition Rates")
    #alpha_rate = st.sidebar.slider('Alpha Rate', min_value=0.1, max_value=1.0, value=0.3, step=0.01)
    alpha_rate = st.sidebar.slider('Alpha Rate',
                               min_value=parameter_ranges['alpha_rate']['min'],
                               max_value=parameter_ranges['alpha_rate']['max'],
                               value=0.1,
                               step=parameter_ranges['alpha_rate']['step'])
    #beta_rate = st.sidebar.slider('Beta Rate', min_value=0.1, max_value=1.0, value=0.5, step=0.01)
    beta_rate = st.sidebar.slider('Beta Rate',
                               min_value=parameter_ranges['beta_rate']['min'],
                               max_value=parameter_ranges['beta_rate']['max'],
                               value=0.2,
                               step=parameter_ranges['beta_rate']['step'])
    #gamma_rate = st.sidebar.slider('Gamma Rate', min_value=0.1, max_value=1.0, value=0.2, step=0.01)
    gamma_rate = st.sidebar.slider('Gamma Rate',
                               min_value=parameter_ranges['gamma_rate']['min'],
                               max_value=parameter_ranges['gamma_rate']['max'],
                               value=0.25,
                               step=parameter_ranges['gamma_rate']['step'])
    
    # Run the optimization
    result = optimize_creatives_pulp(
        budget,
        romi_target,
        cpa_test,
        cpa_boost,
        cpa_scale,
        cpa_alpha,
        conv_test,
        conv_boost,
        conv_scale,
        conv_alpha,
        rev_conv_test,
        rev_conv_boost,
        rev_conv_scale,
        rev_conv_alpha,
        alpha_rate,
        beta_rate,
        gamma_rate
    )
    
    # Display the results
    if result['status'] == 'Optimal':

        parameters = {
            'Budget': budget,
            'ROMI Target': romi_target,
            'CPA Test': cpa_test,
            'CPA Boost': cpa_boost,
            'CPA Scale': cpa_scale,
            'CPA Alpha': cpa_alpha,
            'Conversions per Creative - Test': conv_test,
            'Conversions per Creative - Boost': conv_boost,
            'Conversions per Creative - Scale': conv_scale,
            'Conversions per Creative - Alpha': conv_alpha,
            'Revenue per Conversion - Test': rev_conv_test,
            'Revenue per Conversion - Boost': rev_conv_boost,
            'Revenue per Conversion - Scale': rev_conv_scale,
            'Revenue per Conversion - Alpha': rev_conv_alpha,
            'Alpha Rate': alpha_rate,
            'Beta Rate': beta_rate,
            'Gamma Rate': gamma_rate
        }
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Optimization Results")
            st.write(f"**Number of creatives in Test stage:** {result['n_test']}")
            st.write(f"**Number of creatives in Boost stage:** {result['n_boost']}")
            st.write(f"**Number of creatives in Scale stage:** {result['n_scale']}")
            st.write(f"**Number of creatives in Alpha stage:** {result['n_alpha']}")
            st.write(f"**Total Cost:** ${result['total_cost']:,.2f}")
            st.write(f"**Total Revenue:** ${result['total_revenue']:,.2f}")
            st.write(f"**Actual ROMI:** {result['actual_romi'] * 100:.2f}%")
            
        with col2:
            st.header("Parameter Values")
            
            # Budget and ROMI Target
            #st.subheader("Budget and ROMI Target")
            budget_params = [
                ('Budget', f"${budget:,.2f}"),
                ('ROMI Target', f"{romi_target * 100:.2f}%")
            ]
            for name, value in budget_params:
                st.write(f"- **{name}:** {value}")
            
            # Cost Per Acquisition (CPA)
            #st.subheader("Cost Per Acquisition (CPA)")
            cpa_params = [
                ('CPA Test', f"${cpa_test:,.2f}"),
                ('CPA Boost', f"${cpa_boost:,.2f}"),
                ('CPA Scale', f"${cpa_scale:,.2f}"),
                ('CPA Alpha', f"${cpa_alpha:,.2f}")
            ]
            for name, value in cpa_params:
                st.write(f"- **{name}:** {value}")
            
            # Conversions per Creative
            #st.subheader("Conversions per Creative")
            conv_params = [
                ('Conv Test', conv_test),
                ('Conv Boost', conv_boost),
                ('Conv Scale', conv_scale),
                ('Conv Alpha', conv_alpha)
            ]
            for name, value in conv_params:
                st.write(f"- **{name}:** {value}")
        with col3:
            # Revenue per Conversion
            st.subheader(" ")
            rev_params = [
                ('Rev/Conv Test', f"${rev_conv_test:,.2f}"),
                ('Rev/Conv Boost', f"${rev_conv_boost:,.2f}"),
                ('Rev/Conv Scale', f"${rev_conv_scale:,.2f}"),
                ('Rev/Conv Alpha', f"${rev_conv_alpha:,.2f}")
            ]
            for name, value in rev_params:
                st.write(f"- **{name}:** {value}")
            
            # Transition Rates
            #st.subheader("Transition Rates")
            rate_params = [
                ('Alpha Rate', f"{alpha_rate * 100:.2f}%"),
                ('Beta Rate', f"{beta_rate * 100:.2f}%"),
                ('Gamma Rate', f"{gamma_rate * 100:.2f}%")
            ]
            for name, value in rate_params:
                st.write(f"- **{name}:** {value}")
            

        # Heatmap Analysis
        st.sidebar.header("Heatmap Analysis")
        heatmap_option = st.sidebar.selectbox(
            'Select Heatmap Type',
            ('None', 'Single Parameter Heatmap', 'Two-Parameter Heatmap')
        )




        
        if heatmap_option == 'Single Parameter Heatmap':
            st.sidebar.subheader("Select Parameter to Vary")
            parameter_to_vary = st.sidebar.selectbox(
                'Parameter',
                ('Budget', 'ROMI Target', 'CPA Test', 'CPA Boost', 'CPA Scale', 'CPA Alpha',
                 'Conv Test', 'Conv Boost', 'Conv Scale', 'Conv Alpha',
                 'Rev/Conv Test', 'Rev/Conv Boost', 'Rev/Conv Scale', 'Rev/Conv Alpha',
                 'Alpha Rate', 'Beta Rate', 'Gamma Rate')
            )
            
            param_key = parameter_to_vary.lower().replace(' ', '_').replace('/', '_')
            # Retrieve min, max, and step from parameter_ranges
            param_min = parameter_ranges[param_key]['min']
            param_max = parameter_ranges[param_key]['max']
            param_step = parameter_ranges[param_key]['step']

            st.sidebar.subheader("Parameter Range")
            # Use slider to select the range within the min and max values
            param_range = st.sidebar.slider(f'Select range for {parameter_to_vary}',
                                            min_value=param_min,
                                            max_value=param_max,
                                            value=(param_min, param_max),
                                            step=param_step)
            param_values = np.arange(param_range[0], param_range[1] + param_step, param_step)
            
            romi_values = []
            
            # Loop over parameter values
            for param_value in param_values:
                # Update the parameter dynamically
                kwargs = {
                    'budget': budget,
                    'romi_target': romi_target,
                    'cpa_test': cpa_test,
                    'cpa_boost': cpa_boost,
                    'cpa_scale': cpa_scale,
                    'cpa_alpha': cpa_alpha,
                    'conv_test': conv_test,
                    'conv_boost': conv_boost,
                    'conv_scale': conv_scale,
                    'conv_alpha': conv_alpha,
                    'rev_conv_test': rev_conv_test,
                    'rev_conv_boost': rev_conv_boost,
                    'rev_conv_scale': rev_conv_scale,
                    'rev_conv_alpha': rev_conv_alpha,
                    'alpha_rate': alpha_rate,
                    'beta_rate': beta_rate,
                    'gamma_rate': gamma_rate
                }
                # Set the varying parameter
                param_key = parameter_to_vary.lower().replace(' ', '_').replace('/', '_')
                kwargs[param_key] = param_value
                
                # Run optimization
                result = optimize_creatives_pulp(**kwargs)
                
                # Store ROMI
                if result['status'] == 'Optimal':
                    romi = result['actual_romi']
                else:
                    romi = np.nan  # Assign NaN if optimization failed
                romi_values.append(romi)
            
            # Create a DataFrame for plotting
            heatmap_df = pd.DataFrame({
                parameter_to_vary: param_values,
                'ROMI': romi_values
            })
            
            # Plotting
            st.subheader(f'ROMI vs. {parameter_to_vary}')
            line_chart = alt.Chart(heatmap_df).mark_line().encode(
                x=parameter_to_vary,
                y='ROMI'
            )
            st.altair_chart(line_chart, use_container_width=True)
            pass
        elif heatmap_option == 'Two-Parameter Heatmap':
            st.sidebar.subheader("Select Parameters to Vary")
            param1 = st.sidebar.selectbox(
                'Parameter 1',
                ('Budget', 'ROMI Target', 'CPA Test', 'CPA Boost', 'CPA Scale', 'CPA Alpha',
                 'Conv Test', 'Conv Boost', 'Conv Scale', 'Conv Alpha',
                 'Rev/Conv Test', 'Rev/Conv Boost', 'Rev/Conv Scale', 'Rev/Conv Alpha',
                 'Alpha Rate', 'Beta Rate', 'Gamma Rate'),
                key='param1'
            )
            param2 = st.sidebar.selectbox(
                'Parameter 2',
                ('Budget', 'ROMI Target', 'CPA Test', 'CPA Boost', 'CPA Scale', 'CPA Alpha',
                 'Conv Test', 'Conv Boost', 'Conv Scale', 'Conv Alpha',
                 'Rev/Conv Test', 'Rev/Conv Boost', 'Rev/Conv Scale', 'Rev/Conv Alpha',
                 'Alpha Rate', 'Beta Rate', 'Gamma Rate'),
                index=1,
                key='param2'
            )
            
            # Ensure the parameters are not the same
            if param1 == param2:
                st.error("Please select two different parameters.")
            else:
                # Convert parameter names to keys
                param1_key = param1.lower().replace(' ', '_').replace('/', '_')
                param2_key = param2.lower().replace(' ', '_').replace('/', '_')
                
                # Retrieve min, max, and step from parameter_ranges
                param1_min = parameter_ranges[param1_key]['min']
                param1_max = parameter_ranges[param1_key]['max']
                param1_step = parameter_ranges[param1_key]['step']
                
                param2_min = parameter_ranges[param2_key]['min']
                param2_max = parameter_ranges[param2_key]['max']
                param2_step = parameter_ranges[param2_key]['step']
                
                # Use sliders to select ranges
                param1_range = st.sidebar.slider(f'Select range for {param1}',
                                                 min_value=param1_min,
                                                 max_value=param1_max,
                                                 value=(param1_min, param1_max),
                                                 step=param1_step,
                                                 key='param1_range')
                param2_range = st.sidebar.slider(f'Select range for {param2}',
                                                 min_value=param2_min,
                                                 max_value=param2_max,
                                                 value=(param2_min, param2_max),
                                                 step=param2_step,
                                                 key='param2_range')
                
                # Generate parameter values
                param1_values = np.arange(param1_range[0], param1_range[1] + param1_step, param1_step)
                param2_values = np.arange(param2_range[0], param2_range[1] + param2_step, param2_step)
                
                # Prepare DataFrame to store results
                data = []
                
                for p1 in param1_values:
                    for p2 in param2_values:
                        kwargs = {
                            'budget': budget,
                            'romi_target': romi_target,
                            'cpa_test': cpa_test,
                            'cpa_boost': cpa_boost,
                            'cpa_scale': cpa_scale,
                            'cpa_alpha': cpa_alpha,
                            'conv_test': conv_test,
                            'conv_boost': conv_boost,
                            'conv_scale': conv_scale,
                            'conv_alpha': conv_alpha,
                            'rev_conv_test': rev_conv_test,
                            'rev_conv_boost': rev_conv_boost,
                            'rev_conv_scale': rev_conv_scale,
                            'rev_conv_alpha': rev_conv_alpha,
                            'alpha_rate': alpha_rate,
                            'beta_rate': beta_rate,
                            'gamma_rate': gamma_rate
                        }
                        # Set the varying parameters
                        param1_key = param1.lower().replace(' ', '_').replace('/', '_')
                        param2_key = param2.lower().replace(' ', '_').replace('/', '_')
                        kwargs[param1_key] = p1
                        kwargs[param2_key] = p2
                        
                        # Run optimization
                        result = optimize_creatives_pulp(**kwargs)
                        
                        # Store ROMI
                        if result['status'] == 'Optimal':
                            romi = result['actual_romi']
                        else:
                            romi = np.nan  # Assign NaN if optimization failed
                        data.append({
                            param1: p1,
                            param2: p2,
                            'ROMI': romi
                        })
                
                # Create DataFrame
                heatmap_df = pd.DataFrame(data)
                
                # Plotting
                st.subheader(f'ROMI Heatmap: {param1} vs. {param2}')
                heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
                    x=alt.X(f'{param1}:O', title=param1),
                    y=alt.Y(f'{param2}:O', title=param2),
                    color=alt.Color('ROMI', scale=alt.Scale(scheme='viridis')),
                    tooltip=['ROMI']
                )
                st.altair_chart(heatmap_chart, use_container_width=True)
            pass

        
        # Prepare data frames for plotting
        
        ###stages_df = pd.DataFrame({
        ###    'Stages': result['stages'],
        ###    'Number of Creatives': result['creatives']
        ###}).set_index('Stages')
        
        ###profit_df = pd.DataFrame({
        ###    'Stages': result['stages'],
        ###    'Profit': result['profit_per_stage']
        ###}).set_index('Stages')
        
        ###cost_revenue_df = pd.DataFrame({
        ###    'Amount': [result['total_cost'], result['total_revenue']]
        ###}, index=['Total Cost', 'Total Revenue'])
        
        # Bar Chart for Number of Creatives per Stage
        #st.subheader('Number of Creatives per Stage')
        #st.bar_chart(stages_df)
        
        # Bar Chart for Profit per Stage
        #st.subheader('Profit per Stage')
        #st.bar_chart(profit_df)

        st.header("Parameter Importance Analysis")
        
        if st.button("Run Parameter Importance Analysis"):
            with st.spinner('Generating dataset...'):
                dataset = generate_optimization_dataset(num_samples=100)
            
            st.success("Dataset generated.")
            
            # Prepare data for regression
            X = dataset.drop(columns=['ROMI'])
            y = dataset['ROMI']
            
            # Train regression model
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'Parameter': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Display results
            st.subheader("Parameter Importances")
            st.bar_chart(importance_df.set_index('Parameter'))        

        
    else:
        st.error("Optimization failed.")
        st.error(result['message'])

@st.cache_data
def optimize_creatives_pulp_cached(**kwargs):
    return optimize_creatives_pulp(**kwargs)
    
if __name__ == "__main__":
    main()
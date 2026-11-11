function f = test1(x, model)
    
    % Constants
    pi_value = pi; % MATLAB's built-in constant for pi
    
    % Initialize the output
    f = zeros(size(x, 1), 1); % f will be a column vector with the same number of rows as x
    
    % Loop through each row of x
    for i = 1:size(x, 1)
        x1 = x(i, 1);
        x2 = x(i, 2);
        
        % Calculate the terms
        term1 = (x2 - (5.1 / (4 * pi_value^2)) + (5 / pi_value) * x1 - 6)^2;
        term2 = 10 * ((1 - (1 / (8 * pi_value))) * cos(x1) + 1);
        term3 = (x1 + 5) / 3;
        
        % Calculate the function value for this row
        f(i) = term1 + term2 + term3;
    end
end

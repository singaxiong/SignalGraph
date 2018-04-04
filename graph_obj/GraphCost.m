classdef GraphCost
    properties
        nCost = 1;
        
        totalCost = [];     % total cost of all cost nodes and also regularization cost
        taskCost = [];      % total cost of all cost nodes without regularization cost
        reguCost = [];      % total regularization cost
        subCost = [];       % cost of each cost nodes
        subAcc = [];        % accuracy of cost classification nodes
    end
    
    methods
        function obj = GraphCost(nCost, nEntry)
            if nargin<2; nEntry = 1; end
            obj.totalCost = zeros(1,nEntry);
            obj.taskCost = zeros(nCost,nEntry);
            obj.reguCost = zeros(1,nEntry);
            obj.subCost = zeros(nCost,nEntry);
            obj.subAcc = zeros(nCost,nEntry);
        end
        
        % copy the cost to an outside GraphCost object
        % index specifies the location of the destination. 
        function costDestination = copyCost(obj, costDestination, index)
            if length(obj.totalCost)~=length(index)
                fprintf('GraphCost: Error: the number of destination index is not equal to the number of cost in this object\n');
            end
            costDestination.totalCost(index) = obj.totalCost;
            costDestination.taskCost(:, index) = obj.taskCost;
            costDestination.reguCost(index) = obj.reguCost;
            costDestination.subCost(:, index) = obj.subCost;
            costDestination.subAcc(:, index) = obj.subAcc;
        end
        
        % append the cost of this object to an outside GraphCost object. 
        function costDestination = appendCost(obj, costDestination)
            nEntry = length(obj.totalCost);
            nEntryInDestination = length(costDestination.totalCost);
            index = nEntryInDestination+1 : nEntryInDestination+nEntry;
            costDestination = obj.copyCost(costDestination, index);
        end
        
        function costDestination = appendMeanCost(obj, costDestination)
            costDestination.totalCost(end+1) = mean(obj.totalCost);
            costDestination.taskCost(:, end+1) = mean(obj.taskCost,2);
            costDestination.reguCost(end+1) = mean(obj.reguCost);
            costDestination.subCost(:, end+1) = mean(obj.subCost,2);
            costDestination.subAcc(:, end+1) = mean(obj.subAcc,2);
        end
    end
    
end

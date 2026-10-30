"use strict";

const state = {
    taskType: "classification",
    datasets: [],
    algorithms: [],
    selectedAlgorithms: new Set(),
    parameterValues: {},
    lastBattle: null,
    sessionBattles: 0,
    battleInProgress: false,
};

const API = {
    datasets: "/api/datasets",
    algorithms: "/api/algorithms",
    battle: "/api/battle",
    history: "/api/history",
};

const METRIC_INFO = {
    accuracy: {
        label: "Accuracy",
        direction: "higher",
        description: "Proportion of correct predictions.",
        format: "percent",
    },
    precision: {
        label: "Precision",
        direction: "higher",
        description: "How many positive predictions were correct.",
        format: "decimal",
    },
    recall: {
        label: "Recall",
        direction: "higher",
        description: "How many actual positive cases were found.",
        format: "decimal",
    },
    f1_score: {
        label: "F1 Score",
        direction: "higher",
        description: "Balance between precision and recall.",
        format: "decimal",
    },
    r2: {
        label: "R² Score",
        direction: "higher",
        description: "Explained variation; negative values are possible.",
        format: "decimal",
    },
    mae: {
        label: "Mean Absolute Error",
        direction: "lower",
        description: "Average absolute prediction error.",
        format: "decimal",
    },
    rmse: {
        label: "Root Mean Squared Error",
        direction: "lower",
        description: "Prediction error that penalizes large errors.",
        format: "decimal",
    },
    silhouette_score: {
        label: "Silhouette Score",
        direction: "higher",
        description: "How well-separated the clusters are.",
        format: "decimal",
    },
    calinski_harabasz_score: {
        label: "Calinski–Harabasz Score",
        direction: "higher",
        description: "Cluster separation relative to within-cluster spread.",
        format: "decimal",
    },
    davies_bouldin_score: {
        label: "Davies–Bouldin Score",
        direction: "lower",
        description: "Average cluster similarity; lower is preferable.",
        format: "decimal",
    },
    n_clusters: {
        label: "Number of Clusters",
        direction: "neutral",
        description: "Number of distinct clusters produced.",
        format: "integer",
    },
    training_time: {
        label: "Training Time",
        direction: "lower",
        description: "Time taken to fit the algorithm.",
        format: "time",
    },
};

const PRIMARY_METRIC = {
    classification: "accuracy",
    regression: "r2",
    clustering: "silhouette_score",
};

const $ = (selector) => document.querySelector(selector);

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    })[character]);
}

function formatMetric(value, metricKey) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return String(value);
    }

    const info = METRIC_INFO[metricKey] || {};

    if (info.format === "percent") {
        return `${(number * 100).toFixed(2)}%`;
    }

    if (info.format === "integer") {
        return String(Math.round(number));
    }

    if (info.format === "time") {
        return `${number.toFixed(4)} s`;
    }

    return number.toFixed(4);
}

function metricDirectionText(metricKey) {
    const direction = METRIC_INFO[metricKey]?.direction;

    if (direction === "higher") {
        return "Higher is better";
    }

    if (direction === "lower") {
        return "Lower is better";
    }

    return "Informational metric";
}

function metricLabel(metricKey) {
    return METRIC_INFO[metricKey]?.label
        || metricKey.replaceAll("_", " ");
}

function showMessage(message, type = "info") {
    const container = $("#app-message") || $("#battle-message");

    if (!container) {
        console.log(message);
        return;
    }

    container.textContent = message;
    container.className = `app-message ${type}`;
    container.hidden = false;
}

function clearMessage() {
    const container = $("#app-message") || $("#battle-message");

    if (container) {
        container.textContent = "Configure your experiment before starting.";
        container.hidden = false;
        container.className = "app-message";
    }
}

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
        ...options,
    });

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(`The server returned an unreadable response (${response.status}).`);
    }

    if (!response.ok) {
        throw new Error(data.error || data.message || `Request failed (${response.status}).`);
    }

    return data;
}

function getArrayResponse(data, possibleKeys) {
    for (const key of possibleKeys) {
        if (Array.isArray(data[key])) {
            return data[key];
        }
    }

    return Array.isArray(data) ? data : [];
}

function getDatasetKey(dataset) {
    return String(dataset.key ?? dataset.id ?? dataset.name ?? "");
}

function getAlgorithmKey(algorithm) {
    return String(algorithm.key ?? algorithm.id ?? algorithm.name ?? "");
}

function getAlgorithmName(algorithm) {
    return algorithm.name ?? algorithm.key ?? "Algorithm";
}

function getSelectedDataset() {
    return $("#dataset")?.value || "";
}

function getTaskType() {
    return $("#task-type")?.value || state.taskType;
}

function getSelectedAlgorithms() {
    return [...state.selectedAlgorithms];
}

function renderDatasets() {
    const select = $("#dataset");

    if (!select) {
        return;
    }

    const previousValue = select.value;

    select.replaceChildren();

    for (const dataset of state.datasets) {
        const option = document.createElement("option");
        option.value = getDatasetKey(dataset);
        option.textContent = dataset.name || getDatasetKey(dataset);
        select.appendChild(option);
    }

    if (state.datasets.some((item) => String(getDatasetKey(item)) === previousValue)) {
        select.value = previousValue;
    }

    select.disabled = state.datasets.length === 0;

    const description = $("#dataset-description");
    if (description) {
        description.textContent = state.datasets.length
            ? `${state.datasets.length} dataset(s) available for this task.`
            : "No datasets are available for this task.";
    }
}

function renderAlgorithms() {
    const container = $("#algorithm-options");

    if (!container) {
        return;
    }

    container.replaceChildren();

    for (const algorithm of state.algorithms) {
        const key = getAlgorithmKey(algorithm);
        const label = document.createElement("label");
        label.className = "algorithm-option";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = key;
        checkbox.checked = state.selectedAlgorithms.has(key);
        checkbox.addEventListener("change", () => {
            if (checkbox.checked) {
                state.selectedAlgorithms.add(key);
            } else {
                state.selectedAlgorithms.delete(key);
            }

            renderParameterOptions();
            updateSelectedCount();
        });

        const details = document.createElement("span");
        details.className = "algorithm-option-details";

        const name = document.createElement("strong");
        name.textContent = getAlgorithmName(algorithm);

        const description = document.createElement("small");
        description.textContent = algorithm.description || "Ready to evaluate";

        details.append(name, description);
        label.append(checkbox, details);
        container.appendChild(label);
    }

    updateSelectedCount();
    renderParameterOptions();
}

function updateSelectedCount() {
    const counter = $("#selected-count");

    if (counter) {
        counter.textContent = String(state.selectedAlgorithms.size);
    }
}

function renderParameterOptions() {
    const section = $("#parameter-section");
    const container = $("#parameter-options");

    if (!container) {
        return;
    }

    container.replaceChildren();

    const selected = state.algorithms.filter((algorithm) =>
        state.selectedAlgorithms.has(getAlgorithmKey(algorithm))
    );

    if (section) {
        section.hidden = selected.length === 0;
    }

    for (const algorithm of selected) {
        const key = getAlgorithmKey(algorithm);
        const parameters = Array.isArray(algorithm.parameters)
            ? algorithm.parameters
            : [];

        if (parameters.length === 0) {
            continue;
        }

        const group = document.createElement("section");
        group.className = "parameter-group";

        const heading = document.createElement("h4");
        heading.textContent = getAlgorithmName(algorithm);
        group.appendChild(heading);

        for (const parameter of parameters) {
            const parameterKey = parameter.key;
            const field = document.createElement("label");
            field.className = "parameter-field";

            const title = document.createElement("span");
            title.textContent = parameter.label || parameter.name || parameterKey;
            field.appendChild(title);

            const input = createParameterInput(key, parameter);
            field.appendChild(input);

            if (parameter.description) {
                const help = document.createElement("small");
                help.textContent = parameter.description;
                field.appendChild(help);
            }

            group.appendChild(field);
        }

        container.appendChild(group);
    }
}

function createParameterInput(algorithmKey, parameter) {
    const key = parameter.key;
    const saved = state.parameterValues[algorithmKey]?.[key];
    const defaultValue = saved ?? parameter.default ?? "";

    let input;

    if (Array.isArray(parameter.options) && parameter.options.length > 0) {
        input = document.createElement("select");

        for (const optionValue of parameter.options) {
            const option = document.createElement("option");

            if (typeof optionValue === "object" && optionValue !== null) {
                option.value = optionValue.value;
                option.textContent = optionValue.label ?? optionValue.value;
            } else {
                option.value = optionValue;
                option.textContent = optionValue;
            }

            input.appendChild(option);
        }

        input.value = defaultValue;
    } else {
        input = document.createElement("input");

        const type = String(parameter.type || "").toLowerCase();

        if (["integer", "int"].includes(type)) {
            input.type = "number";
            input.step = "1";
        } else if (["number", "float", "double"].includes(type)) {
            input.type = "number";
            input.step = "any";
        } else {
            input.type = "text";
        }

        if (parameter.min !== undefined) {
            input.min = parameter.min;
        }

        if (parameter.max !== undefined) {
            input.max = parameter.max;
        }

        input.value = defaultValue;
    }

    input.name = `${algorithmKey}__${key}`;
    input.dataset.algorithm = algorithmKey;
    input.dataset.parameter = key;
    input.addEventListener("change", () => {
        const currentAlgorithm = input.dataset.algorithm;
        const currentParameter = input.dataset.parameter;

        state.parameterValues[currentAlgorithm] ||= {};
        state.parameterValues[currentAlgorithm][currentParameter] = input.value;
    });

    return input;
}

function collectAlgorithmParameters() {
    const result = {};

    for (const input of document.querySelectorAll(
        "#parameter-options [data-algorithm][data-parameter]"
    )) {
        const algorithmKey = input.dataset.algorithm;
        const parameterKey = input.dataset.parameter;
        const algorithm = state.algorithms.find(
            (item) => getAlgorithmKey(item) === algorithmKey
        );

        const schema = algorithm?.parameters?.find(
            (item) => item.key === parameterKey
        );

        let value = input.value;

        if (schema) {
            const type = String(schema.type || "").toLowerCase();

            if (["integer", "int", "number", "float", "double"].includes(type)) {
                if (value.trim() === "") {
                    throw new Error(`Enter a value for ${parameterKey}.`);
                }

                value = Number(value);

                if (!Number.isFinite(value)) {
                    throw new Error(`${parameterKey} must be a valid number.`);
                }

                if (["integer", "int"].includes(type) && !Number.isInteger(value)) {
                    throw new Error(`${parameterKey} must be a whole number.`);
                }

                if (schema.min !== undefined && value < schema.min) {
                    throw new Error(`${parameterKey} must be at least ${schema.min}.`);
                }

                if (schema.max !== undefined && value > schema.max) {
                    throw new Error(`${parameterKey} must not exceed ${schema.max}.`);
                }
            }
        }

        result[algorithmKey] ||= {};
        result[algorithmKey][parameterKey] = value;
    }

    return result;
}

async function loadDatasets() {
    const data = await fetchJSON(
        `${API.datasets}?task_type=${encodeURIComponent(state.taskType)}`
    );

    state.datasets = getArrayResponse(data, ["datasets", "items"]);
    renderDatasets();
}

async function loadAlgorithms() {
    const data = await fetchJSON(
        `${API.algorithms}?task_type=${encodeURIComponent(state.taskType)}`
    );

    state.algorithms = getArrayResponse(data, ["algorithms", "items"]);

    const validKeys = new Set(state.algorithms.map(getAlgorithmKey));

    state.selectedAlgorithms = new Set(
        [...state.selectedAlgorithms].filter((key) => validKeys.has(key))
    );

    if (state.selectedAlgorithms.size === 0 && state.algorithms.length > 0) {
        state.selectedAlgorithms.add(getAlgorithmKey(state.algorithms[0]));
    }

    renderAlgorithms();
}

async function handleTaskChange() {
    clearMessage();

    state.taskType = getTaskType();
    state.datasets = [];
    state.algorithms = [];
    state.selectedAlgorithms.clear();
    state.parameterValues = {};

    try {
        await Promise.all([loadDatasets(), loadAlgorithms()]);
    } catch (error) {
        showMessage(error.message, "error");
    }
}

function getMetricEntries(metrics) {
    return Object.entries(metrics || {}).filter(([key, value]) =>
        typeof value === "number" && Number.isFinite(value)
        && key !== "confusion_matrix"
    );
}

function compareMetricValues(a, b, metricKey) {
    const direction = METRIC_INFO[metricKey]?.direction || "neutral";

    if (direction === "lower") {
        return a - b;
    }

    if (direction === "higher") {
        return b - a;
    }

    return 0;
}

function getBestResultIndex(results, metricKey) {
    const info = METRIC_INFO[metricKey];

    if (!info || info.direction === "neutral") {
        return -1;
    }

    let bestIndex = -1;
    let bestValue = null;

    results.forEach((result, index) => {
        const value = result.metrics?.[metricKey];

        if (typeof value !== "number" || !Number.isFinite(value)) {
            return;
        }

        if (bestIndex === -1) {
            bestIndex = index;
            bestValue = value;
            return;
        }

        const comparison = compareMetricValues(value, bestValue, metricKey);

        if (comparison < 0) {
            bestIndex = index;
            bestValue = value;
        }
    });

    return bestIndex;
}

function getWinnerName(battle) {
    if (typeof battle.winner === "string") {
        return battle.winner;
    }

    if (battle.winner && typeof battle.winner === "object") {
        return battle.winner.name
            || battle.winner.algorithm
            || battle.winner.algorithm_name
            || "Winner";
    }

    return battle.winner_name || "";
}

function createMetricChart(results, metricKey) {
    const section = document.createElement("section");
    section.className = "metric-chart-card";

    const info = METRIC_INFO[metricKey] || {};
    const title = document.createElement("h4");
    title.textContent = metricLabel(metricKey);

    const direction = document.createElement("span");
    direction.className = `metric-direction direction-${info.direction || "neutral"}`;
    direction.textContent = metricDirectionText(metricKey);

    const description = document.createElement("p");
    description.className = "metric-description";
    description.textContent = info.description || "Evaluation metric.";

    const heading = document.createElement("div");
    heading.className = "metric-chart-heading";
    heading.append(title, direction);

    const rows = results
        .map((result, index) => ({
            result,
            index,
            value: result.metrics?.[metricKey],
        }))
        .filter((item) =>
            typeof item.value === "number" && Number.isFinite(item.value)
        );

    const list = document.createElement("div");
    list.className = "metric-bars";

    if (rows.length === 0) {
        description.textContent = "No numeric values are available for this metric.";
        section.append(heading, description, list);
        return section;
    }

    const values = rows.map((row) => row.value);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const span = maximum - minimum;
    const bestIndex = getBestResultIndex(results, metricKey);

    for (const row of rows) {
        const item = document.createElement("div");
        item.className = "metric-bar-row";

        if (row.index === bestIndex) {
            item.classList.add("metric-best");
        }

        const label = document.createElement("div");
        label.className = "metric-bar-label";
        label.textContent = row.result.name || "Algorithm";

        if (row.index === bestIndex) {
            const badge = document.createElement("span");
            badge.className = "metric-best-badge";
            badge.textContent = "BEST";
            label.appendChild(badge);
        }

        const track = document.createElement("div");
        track.className = "metric-bar-track";

        const bar = document.createElement("div");
        bar.className = "metric-bar-fill";

        // Normalize values to the observed range, not to a fixed
        // percentage scale. This avoids treating unrelated metrics
        // as if they shared the same units or scale.
        let width = 100;

        if (span > 0) {
            width = ((row.value - minimum) / span) * 85 + 15;
        }

        bar.style.width = `${width}%`;

        if (row.index === bestIndex) {
            bar.classList.add("is-best");
        }

        track.appendChild(bar);

        const value = document.createElement("span");
        value.className = "metric-bar-value";
        value.textContent = formatMetric(row.value, metricKey);

        item.append(label, track, value);
        list.appendChild(item);
    }

    const note = document.createElement("p");
    note.className = "chart-scale-note";
    note.textContent =
        "Bars show relative values within this metric only. " +
        "Compare values and use the direction label to identify the better result.";

    section.append(heading, description, list, note);

    return section;
}

function renderMetricCharts(container, results) {
    const metrics = new Set();

    for (const result of results) {
        for (const [key] of getMetricEntries(result.metrics)) {
            metrics.add(key);
        }
    }

    const grid = document.createElement("div");
    grid.className = "metric-chart-grid";

    for (const key of metrics) {
        grid.appendChild(createMetricChart(results, key));
    }

    container.appendChild(grid);
}

function renderLeaderboard(container, results, taskType) {
    const metricKey = PRIMARY_METRIC[taskType];

    if (!metricKey) {
        return;
    }

    const sorted = results
        .filter((result) => Number.isFinite(result.metrics?.[metricKey]))
        .slice()
        .sort((a, b) => compareMetricValues(
            a.metrics[metricKey],
            b.metrics[metricKey],
            metricKey
        ));

    const section = document.createElement("section");
    section.className = "leaderboard-section";

    const heading = document.createElement("h3");
    heading.textContent = "Algorithm Leaderboard";

    const description = document.createElement("p");
    description.textContent =
        `Ranked by ${metricLabel(metricKey)}. ${metricDirectionText(metricKey)}.`;

    const table = document.createElement("table");
    table.className = "leaderboard-table";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    for (const label of ["Rank", "Algorithm", metricLabel(metricKey), "Status"]) {
        const th = document.createElement("th");
        th.textContent = label;
        headerRow.appendChild(th);
    }

    thead.appendChild(headerRow);

    const tbody = document.createElement("tbody");

    sorted.forEach((result, index) => {
        const row = document.createElement("tr");

        if (index === 0) {
            row.className = "leaderboard-winner";
        }

        const rank = document.createElement("td");
        rank.textContent = `#${index + 1}`;

        const name = document.createElement("td");
        name.textContent = result.name || "Algorithm";

        const metric = document.createElement("td");
        metric.textContent = formatMetric(result.metrics[metricKey], metricKey);

        const status = document.createElement("td");
        status.textContent = index === 0 ? "TOP RESULT" : "Evaluated";

        if (index === 0) {
            status.className = "winner-status";
        }

        row.append(rank, name, metric, status);
        tbody.appendChild(row);
    });

    table.append(thead, tbody);
    section.append(heading, description, table);
    container.appendChild(section);
}

function renderConfusionMatrix(container, result) {
    const matrix = result.metrics?.confusion_matrix;

    if (!Array.isArray(matrix) || matrix.length === 0) {
        return;
    }

    const section = document.createElement("section");
    section.className = "confusion-matrix-section";

    const heading = document.createElement("h4");
    heading.textContent = `${result.name || "Algorithm"} — Confusion Matrix`;

    const table = document.createElement("table");
    table.className = "confusion-matrix";

    matrix.forEach((rowValues, rowIndex) => {
        const row = document.createElement("tr");

        rowValues.forEach((value, columnIndex) => {
            const cell = document.createElement(
                rowIndex === 0 ? "th" : "td"
            );

            cell.textContent = String(value);

            if (rowIndex === columnIndex) {
                cell.className = "matrix-correct";
            }

            row.appendChild(cell);
        });

        table.appendChild(row);
    });

    section.append(heading, table);
    container.appendChild(section);
}

function renderResults(battle) {
    const container = $("#battle-results");

    if (!container) {
        return;
    }

    container.replaceChildren();

    const results = Array.isArray(battle.results) ? battle.results : [];

    if (results.length === 0) {
        container.textContent = "The battle returned no algorithm results.";
        return;
    }

    const heading = document.createElement("h2");
    heading.textContent = "Battle Results";

    const summary = document.createElement("p");
    summary.className = "battle-summary";
    summary.textContent =
        `${battle.dataset_name || battle.dataset || "Dataset"} · ` +
        `${battle.task_type || state.taskType} · ` +
        `${results.length} algorithm(s) evaluated`;

    container.append(heading, summary);

    const winnerName = getWinnerName(battle);

    if (winnerName) {
        const winner = document.createElement("div");
        winner.className = "battle-winner";

        const label = document.createElement("span");
        label.className = "winner-eyebrow";
        label.textContent = "BATTLE WINNER";

        const name = document.createElement("h3");
        name.textContent = winnerName;

        const metricKey = PRIMARY_METRIC[battle.task_type || state.taskType];
        const winnerResult = results.find((result) =>
            result.name === winnerName
            || result.key === winnerName
        );

        const detail = document.createElement("p");

        if (winnerResult && metricKey) {
            detail.textContent =
                `${metricLabel(metricKey)}: ` +
                `${formatMetric(winnerResult.metrics?.[metricKey], metricKey)} · ` +
                `${metricDirectionText(metricKey)}`;
        } else {
            detail.textContent =
                "Winner selected using the task's primary evaluation metric.";
        }

        winner.append(label, name, detail);
        container.appendChild(winner);
    }

    renderLeaderboard(container, results, battle.task_type || state.taskType);

    const chartsHeading = document.createElement("h3");
    chartsHeading.textContent = "Metric Comparison";
    container.appendChild(chartsHeading);

    renderMetricCharts(container, results);

    const tableHeading = document.createElement("h3");
    tableHeading.textContent = "Detailed Results";
    container.appendChild(tableHeading);

    const table = document.createElement("table");
    table.className = "results-table";

    const metricKeys = [...new Set(
        results.flatMap((result) =>
            getMetricEntries(result.metrics).map(([key]) => key)
        )
    )];

    const headers = [
        "Algorithm",
        ...metricKeys.map(metricLabel),
        "Training Time",
        "Parameters",
    ];

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach((label) => {
        const th = document.createElement("th");
        th.textContent = label;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);

    const tbody = document.createElement("tbody");

    results.forEach((result) => {
        const row = document.createElement("tr");

        const name = document.createElement("td");
        name.textContent = result.name || "Algorithm";
        row.appendChild(name);

        metricKeys.forEach((key) => {
            const cell = document.createElement("td");
            cell.textContent = formatMetric(result.metrics?.[key], key);
            cell.title = metricDirectionText(key);
            row.appendChild(cell);
        });

        const timeCell = document.createElement("td");
        const time = result.training_time ?? result.fit_time ?? null;
        timeCell.textContent = time === null
            ? "—"
            : `${Number(time).toFixed(4)} s`;
        row.appendChild(timeCell);

        const parameterCell = document.createElement("td");
        parameterCell.textContent = JSON.stringify(result.parameters || {});
        row.appendChild(parameterCell);

        tbody.appendChild(row);
    });

    table.append(thead, tbody);
    container.appendChild(table);

    const matrixContainer = document.createElement("div");
    matrixContainer.className = "confusion-matrices";

    results.forEach((result) => {
        renderConfusionMatrix(matrixContainer, result);
    });

    if (matrixContainer.childElementCount > 0) {
        container.appendChild(matrixContainer);
    }

    const note = document.createElement("p");
    note.className = "evaluation-note";
    note.textContent =
        "The overall winner is chosen using the primary metric for the selected task. " +
        "Different metrics measure different properties, so a winner on one metric " +
        "does not necessarily lead on every other metric.";

    container.appendChild(note);
}

async function runBattle(event) {
    event?.preventDefault();

    // Ignore accidental duplicate click + submit events.
    if (state.battleInProgress) {
        return;
    }

    clearMessage();

    const dataset = getSelectedDataset();
    const taskType = getTaskType();
    const algorithms = getSelectedAlgorithms();

    if (!dataset) {
        showMessage("Select a dataset before running a battle.", "error");
        return;
    }

    if (algorithms.length < 2) {
        showMessage("Select at least two algorithms to compare.", "error");
        return;
    }

    let algorithmParams;

    try {
        algorithmParams = collectAlgorithmParameters();
    } catch (error) {
        showMessage(error.message, "error");
        return;
    }

    state.battleInProgress = true;
    const button = $("#configure-battle");
    const originalText = button?.textContent;

    if (button) {
        button.disabled = true;
        button.textContent = "Running battle...";
    }

    try {
        const testSizeInput = $("#test-size");
        const testSize = testSizeInput
            ? Number(testSizeInput.value)
            : 0.2;

        const payload = {
            dataset,
            task_type: taskType,
            algorithms,
            test_size: Number.isFinite(testSize) ? testSize : 0.2,
            random_state: 42,
            algorithm_params: algorithmParams,
        };

        const battle = await fetchJSON(API.battle, {
            method: "POST",
            body: JSON.stringify(payload),
        });

        state.lastBattle = battle;
        state.sessionBattles += 1;

        const counter = $("#comparison-count") || $("#session-count");

        if (counter) {
            counter.textContent = String(state.sessionBattles);
        }

        renderResults(battle);
        await loadHistory();

        showMessage("Battle completed successfully.", "success");
    } catch (error) {
        showMessage(error.message, "error");
    } finally {
        state.battleInProgress = false;
        if (button) {
            button.disabled = false;
            button.textContent = originalText || "Run Battle";
        }
    }
}

function renderHistory(battles) {
    const container = $("#recent-history");

    if (!container) {
        return;
    }

    container.replaceChildren();

    if (!battles.length) {
        container.textContent = "No previous battles found.";
        return;
    }

    for (const battle of battles) {
        const item = document.createElement("article");
        item.className = "history-item";

        const title = document.createElement("h4");
        title.textContent =
            `${battle.dataset_name || battle.dataset || "Dataset"} · ` +
            `${battle.task_type || "Experiment"}`;

        const details = document.createElement("p");
        details.textContent =
            `${battle.created_at || battle.timestamp || "Saved battle"} · ` +
            `${Array.isArray(battle.results) ? battle.results.length : 0} results`;

        item.append(title, details);

        if (Array.isArray(battle.results) && battle.results.length > 0) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "button button-secondary";
            button.textContent = "View Results";

            button.addEventListener("click", () => {
                const saved = {
                    ...battle,
                    task_type: battle.task_type || state.taskType,
                };

                state.lastBattle = saved;
                renderResults(saved);

                $("#battle-results")?.scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                });
            });

            item.appendChild(button);
        }

        container.appendChild(item);
    }
}

async function loadHistory() {
    const container = $("#recent-history");

    if (!container) {
        return;
    }

    try {
        const data = await fetchJSON(`${API.history}?limit=10`);
        const battles = getArrayResponse(data, ["battles", "history", "items"]);
        renderHistory(battles);
    } catch (error) {
        container.textContent = `Could not load history: ${error.message}`;
    }
}

function attachEventListeners() {
    $("#task-type")?.addEventListener("change", handleTaskChange);
    const battleForm = $("#battle-form");
    const battleButton = $("#configure-battle");

    // Support both clicking the button and submitting the form. Prevent the
    // browser's native submit after the click handler starts the battle.
    battleButton?.addEventListener("click", (event) => {
        event.preventDefault();
        runBattle(event);
    });
    battleForm?.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!battleButton || !battleButton.disabled) {
            runBattle(event);
        }
    });

    $("#select-all-algorithms")?.addEventListener("click", () => {
        state.algorithms.forEach((algorithm) => {
            state.selectedAlgorithms.add(getAlgorithmKey(algorithm));
        });

        renderAlgorithms();
    });

    $("#clear-algorithms")?.addEventListener("click", () => {
        state.selectedAlgorithms.clear();
        renderAlgorithms();
    });
}

async function initializeApp() {
    attachEventListeners();

    const taskSelect = $("#task-type");
    if (taskSelect && !taskSelect.value) {
        taskSelect.value = "classification";
    }
    state.taskType = getTaskType() || "classification";

    try {
        await Promise.all([loadDatasets(), loadAlgorithms()]);
        await loadHistory();
    } catch (error) {
        showMessage(error.message, "error");
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeApp, { once: true });
} else {
    initializeApp();
}
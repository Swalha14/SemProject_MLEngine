
package com.example.elimisha

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.*
import kotlin.math.*
import kotlin.random.Random

private val ForestGreen = Color(0xFF0E4100)
private val LightGreen = Color(0xFFE0F7E9)
private val PaleBackground = Color(0xFFF5F8F4)
private val DarkText = Color(0xFF18231A)
private val MutedText = Color(0xFF718071)
private val BorderColor = Color(0xFFE1E9E0)
private val White = Color.White
private val WarningColor = Color(0xFFB7791F)
private val RedColor = Color(0xFFB42318)

private fun formatNumber(value: Double, decimals: Int = 2): String {
    val multiplier = when (decimals) {
        1 -> 10.0
        2 -> 100.0
        3 -> 1000.0
        else -> 100.0
    }

    val rounded = round(value * multiplier) / multiplier
    val text = rounded.toString()
    val parts = text.split(".")
    val integerPart = parts[0]
    val decimalPart = parts.getOrElse(1) { "" }
        .padEnd(decimals, '0')
        .take(decimals)

    return "$integerPart.$decimalPart"
}

// ------------------------------------------------------------
// STUDENT DATA
// ------------------------------------------------------------

data class StudentRecord(
    val id: Int,
    val g1: Int,
    val g2: Int,
    val studyTime: Int,
    val failures: Int,
    val absences: Int,
    val activities: Boolean,
    val romantic: Boolean,
    val g3: Double
)

data class StudentInput(
    val g1: Double,
    val g2: Double,
    val studyTime: Double,
    val failures: Double,
    val absences: Double,
    val activities: Boolean,
    val romantic: Boolean
)

// ------------------------------------------------------------
// DUMMY DATASET
// ------------------------------------------------------------

object DummyDataset {

    fun generate(count: Int = 395): List<StudentRecord> {

        val random = Random(42)

        return List(count) { index ->

            val g1 = random.nextInt(0, 21)
            val g2 = random.nextInt(0, 21)
            val studyTime = random.nextInt(1, 5)

            val failures = when (random.nextInt(100)) {
                in 0..66 -> 0
                in 67..86 -> 1
                in 87..95 -> 2
                else -> 3
            }

            val absences = when (random.nextInt(100)) {
                in 0..29 -> random.nextInt(0, 2)
                in 30..64 -> random.nextInt(2, 6)
                in 65..89 -> random.nextInt(6, 12)
                else -> random.nextInt(12, 31)
            }

            val activities = random.nextBoolean()
            val romantic = random.nextBoolean()

            val noise = random.nextDouble(-2.2, 2.2)

            val grade =
                (0.12 * g1) +
                        (0.76 * g2) +
                        (0.22 * studyTime) -
                        (0.25 * failures) -
                        (0.035 * absences) +
                        (if (activities) 0.15 else 0.0) +
                        (if (romantic) -0.10 else 0.0) +
                        noise

            StudentRecord(
                id = index + 1,
                g1 = g1,
                g2 = g2,
                studyTime = studyTime,
                failures = failures,
                absences = absences,
                activities = activities,
                romantic = romantic,
                g3 = grade.coerceIn(0.0, 20.0)
            )
        }
    }
}

// ------------------------------------------------------------
// FEATURE ENCODING
// ------------------------------------------------------------

object FeatureEncoder {

    val featureNames = listOf(
        "G1",
        "G2",
        "Study time",
        "Failures",
        "Absences",
        "Activities",
        "Romantic"
    )

    fun encode(input: StudentInput): DoubleArray {
        return doubleArrayOf(
            input.g1,
            input.g2,
            input.studyTime,
            input.failures,
            input.absences,
            if (input.activities) 1.0 else 0.0,
            if (input.romantic) 1.0 else 0.0
        )
    }

    fun encode(record: StudentRecord): DoubleArray {
        return encode(
            StudentInput(
                g1 = record.g1.toDouble(),
                g2 = record.g2.toDouble(),
                studyTime = record.studyTime.toDouble(),
                failures = record.failures.toDouble(),
                absences = record.absences.toDouble(),
                activities = record.activities,
                romantic = record.romantic
            )
        )
    }
}

// ------------------------------------------------------------
// INTERNAL PREDICTION TREE
// ------------------------------------------------------------

private data class TreeNode(
    val prediction: Double,
    val feature: Int = -1,
    val threshold: Double = 0.0,
    val left: TreeNode? = null,
    val right: TreeNode? = null
) {

    fun predict(features: DoubleArray): Double {

        if (feature == -1) return prediction

        return if (features[feature] <= threshold) {
            left!!.predict(features)
        } else {
            right!!.predict(features)
        }
    }
}

private data class Split(
    val feature: Int,
    val threshold: Double,
    val score: Double
)

// ------------------------------------------------------------
// INTERNAL GRADE PREDICTION ENGINE
// ------------------------------------------------------------

class RandomForestRegressor(
    private val numberOfTrees: Int = 100,
    private val maxDepth: Int = 8,
    private val minSamples: Int = 4,
    private val seed: Int = 42
) {

    private val trees = mutableListOf<TreeNode>()
    private val featureImportance = DoubleArray(7)
    private val random = Random(seed)

    fun train(
        features: List<DoubleArray>,
        targets: List<Double>
    ) {

        require(features.isNotEmpty()) {
            "Training data cannot be empty."
        }

        require(features.size == targets.size) {
            "Feature and target sizes must match."
        }

        require(features.all { it.size == 7 }) {
            "Every record must contain exactly seven features."
        }

        trees.clear()
        featureImportance.fill(0.0)

        repeat(numberOfTrees) {

            val sampledIndices = List(features.size) {
                random.nextInt(features.size)
            }

            val sampledFeatures = sampledIndices.map {
                features[it]
            }

            val sampledTargets = sampledIndices.map {
                targets[it]
            }

            trees.add(
                buildTree(
                    sampledFeatures,
                    sampledTargets,
                    depth = 0
                )
            )
        }

        val totalImportance = featureImportance.sum()

        if (totalImportance > 0.0) {
            for (i in featureImportance.indices) {
                featureImportance[i] /= totalImportance
            }
        }
    }

    fun predict(features: DoubleArray): Double {

        require(trees.isNotEmpty()) {
            "The prediction engine has not been initialized."
        }

        require(features.size == 7) {
            "Prediction requires exactly seven features."
        }

        return trees.sumOf {
            it.predict(features)
        } / trees.size
    }

    fun importance(): List<Pair<String, Double>> {

        return FeatureEncoder.featureNames
            .mapIndexed { index, name ->
                name to featureImportance[index]
            }
            .sortedByDescending { it.second }
    }

    private fun buildTree(
        x: List<DoubleArray>,
        y: List<Double>,
        depth: Int
    ): TreeNode {

        val mean = y.average()

        if (
            depth >= maxDepth ||
            x.size < minSamples ||
            y.distinct().size <= 1
        ) {
            return TreeNode(mean)
        }

        val featureCount = max(
            1,
            sqrt(x.first().size.toDouble()).toInt()
        )

        val selectedFeatures = (0 until x.first().size)
            .shuffled(random)
            .take(featureCount)

        val split = findBestSplit(
            x,
            y,
            selectedFeatures
        ) ?: return TreeNode(mean)

        val leftIndices = x.indices.filter {
            x[it][split.feature] <= split.threshold
        }

        val rightIndices = x.indices.filter {
            x[it][split.feature] > split.threshold
        }

        if (
            leftIndices.size < minSamples ||
            rightIndices.size < minSamples
        ) {
            return TreeNode(mean)
        }

        val leftY = leftIndices.map { y[it] }
        val rightY = rightIndices.map { y[it] }

        val parentVariance = variance(y)

        val childVariance =
            (leftY.size * variance(leftY) +
                    rightY.size * variance(rightY)) / y.size

        featureImportance[split.feature] +=
            max(0.0, parentVariance - childVariance) * y.size

        val leftNode = buildTree(
            leftIndices.map { x[it] },
            leftY,
            depth + 1
        )

        val rightNode = buildTree(
            rightIndices.map { x[it] },
            rightY,
            depth + 1
        )

        return TreeNode(
            prediction = mean,
            feature = split.feature,
            threshold = split.threshold,
            left = leftNode,
            right = rightNode
        )
    }

    private fun findBestSplit(
        x: List<DoubleArray>,
        y: List<Double>,
        selectedFeatures: List<Int>
    ): Split? {

        var bestSplit: Split? = null
        val parentScore = variance(y) * y.size

        for (feature in selectedFeatures) {

            val values = x
                .map { it[feature] }
                .distinct()
                .sorted()

            if (values.size <= 1) continue

            val thresholds = values.zipWithNext { a, b ->
                (a + b) / 2.0
            }

            for (threshold in thresholds) {

                val left = mutableListOf<Double>()
                val right = mutableListOf<Double>()

                for (i in x.indices) {
                    if (x[i][feature] <= threshold) {
                        left.add(y[i])
                    } else {
                        right.add(y[i])
                    }
                }

                if (left.isEmpty() || right.isEmpty()) continue

                val score =
                    left.size * variance(left) +
                            right.size * variance(right)

                if (
                    score < parentScore &&
                    (bestSplit == null || score < bestSplit.score)
                ) {
                    bestSplit = Split(
                        feature = feature,
                        threshold = threshold,
                        score = score
                    )
                }
            }
        }

        return bestSplit
    }

    private fun variance(values: List<Double>): Double {

        if (values.isEmpty()) return 0.0

        val mean = values.average()

        return values.sumOf {
            (it - mean).pow(2)
        } / values.size
    }
}

// ------------------------------------------------------------
// STUDENT PERFORMANCE ANALYTICS
// ------------------------------------------------------------

data class StudentAnalytics(
    val totalStudents: Int,
    val averageGrade: Double,
    val highestGrade: Double,
    val lowestGrade: Double,
    val passCount: Int,
    val failCount: Int,
    val passRate: Double,
    val excellentCount: Int,
    val goodCount: Int,
    val averageCount: Int,
    val belowAverageCount: Int,
    val atRiskCount: Int
)

object StudentAnalyticsCalculator {

    fun calculate(records: List<StudentRecord>): StudentAnalytics {

        require(records.isNotEmpty()) {
            "Student records cannot be empty."
        }

        val grades = records.map { it.g3 }

        val passCount = grades.count { it >= 10.0 }

        return StudentAnalytics(
            totalStudents = records.size,
            averageGrade = grades.average(),
            highestGrade = grades.maxOrNull() ?: 0.0,
            lowestGrade = grades.minOrNull() ?: 0.0,
            passCount = passCount,
            failCount = records.size - passCount,
            passRate = passCount.toDouble() / records.size * 100.0,
            excellentCount = grades.count { it >= 16.0 },
            goodCount = grades.count { it >= 12.0 && it < 16.0 },
            averageCount = grades.count { it >= 10.0 && it < 12.0 },
            belowAverageCount = grades.count { it >= 7.0 && it < 10.0 },
            atRiskCount = grades.count { it < 7.0 }
        )
    }
}

// ------------------------------------------------------------
// MODEL EVALUATION - INTERNAL ONLY
// ------------------------------------------------------------

data class ModelMetrics(
    val mae: Double,
    val rmse: Double,
    val r2: Double
)

fun evaluateModel(
    actual: List<Double>,
    predicted: List<Double>
): ModelMetrics {

    require(actual.isNotEmpty())
    require(actual.size == predicted.size)

    val errors = actual.indices.map {
        actual[it] - predicted[it]
    }

    val mae = errors.map { abs(it) }.average()
    val mse = errors.map { it * it }.average()
    val rmse = sqrt(mse)

    val mean = actual.average()

    val totalSumSquares = actual.sumOf {
        (it - mean).pow(2)
    }

    val residualSumSquares = errors.sumOf {
        it * it
    }

    val r2 = if (totalSumSquares == 0.0) {
        0.0
    } else {
        1.0 - residualSumSquares / totalSumSquares
    }

    return ModelMetrics(
        mae = mae,
        rmse = rmse,
        r2 = r2
    )
}

// ------------------------------------------------------------
// TRAINED PREDICTION ENGINE
// ------------------------------------------------------------

data class TrainedGradeModel(
    val forest: RandomForestRegressor,
    val metrics: ModelMetrics,
    val records: List<StudentRecord>,
    val trainingSize: Int,
    val testingSize: Int
)

object GradeModelTrainer {

    fun train(records: List<StudentRecord>): TrainedGradeModel {

        require(records.size >= 10) {
            "At least ten records are required."
        }

        val shuffled = records.shuffled(Random(42))
        val splitIndex = (shuffled.size * 0.8).toInt()

        val trainingData = shuffled.take(splitIndex)
        val testingData = shuffled.drop(splitIndex)

        val forest = RandomForestRegressor(
            numberOfTrees = 100,
            maxDepth = 8,
            minSamples = 4,
            seed = 42
        )

        forest.train(
            features = trainingData.map {
                FeatureEncoder.encode(it)
            },
            targets = trainingData.map {
                it.g3
            }
        )

        val actual = testingData.map {
            it.g3
        }

        val predicted = testingData.map {
            forest.predict(
                FeatureEncoder.encode(it)
            ).coerceIn(0.0, 20.0)
        }

        return TrainedGradeModel(
            forest = forest,
            metrics = evaluateModel(actual, predicted),
            records = records,
            trainingSize = trainingData.size,
            testingSize = testingData.size
        )
    }
}

// ------------------------------------------------------------
// MAIN APPLICATION
// ------------------------------------------------------------

@Composable
fun App() {
    MainScreen()
}

@Composable
fun MainScreen() {

    val records = remember {
        DummyDataset.generate(395)
    }

    val trainedModel = remember {
        GradeModelTrainer.train(records)
    }

    val analytics = remember(records) {
        StudentAnalyticsCalculator.calculate(records)
    }

    var selectedTab by remember {
        mutableStateOf("Dashboard")
    }

    MaterialTheme {

        Surface(
            modifier = Modifier.fillMaxSize(),
            color = PaleBackground
        ) {

            Column(
                modifier = Modifier.fillMaxSize()
            ) {

                Header()

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {

                    listOf(
                        "Dashboard",
                        "Marklist",
                        "Predict Grade"
                    ).forEach { tab ->

                        FilterChip(
                            selected = selectedTab == tab,
                            onClick = {
                                selectedTab = tab
                            },
                            label = {
                                Text(tab)
                            },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = ForestGreen,
                                selectedLabelColor = White
                            )
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))

                when (selectedTab) {

                    "Dashboard" -> DashboardScreen(
                        analytics = analytics
                    )

                    "Marklist" -> MarklistScreen(records)

                    "Predict Grade" -> PredictionScreen(
                        model = trainedModel.forest
                    )
                }
            }
        }
    }
}

// ------------------------------------------------------------
// HEADER
// ------------------------------------------------------------

@Composable
fun Header() {

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(White)
            .padding(20.dp)
    ) {

        Text(
            "ELIVIA",
            color = ForestGreen,
            fontWeight = FontWeight.ExtraBold,
            fontSize = 13.sp
        )

        Spacer(Modifier.height(5.dp))

        Text(
            "Mathematics Performance",
            color = DarkText,
            fontWeight = FontWeight.Bold,
            fontSize = 23.sp
        )

        Text(
            "Student analysis, academic insights and grade prediction",
            color = MutedText,
            fontSize = 13.sp
        )
    }
}

// ------------------------------------------------------------
// DASHBOARD
// ------------------------------------------------------------

@Composable
fun DashboardScreen(analytics: StudentAnalytics) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        Text(
            "Overview",
            fontSize = 21.sp,
            fontWeight = FontWeight.Bold,
            color = DarkText
        )

        Text(
            "A summary of the Mathematics performance of all students.",
            color = MutedText,
            fontSize = 13.sp
        )

        // TOTAL STUDENTS

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(15.dp),
            colors = CardDefaults.cardColors(
                containerColor = ForestGreen
            )
        ) {

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {

                Column {

                    Text(
                        "Total Students",
                        color = Color.White.copy(alpha = 0.8f),
                        fontSize = 13.sp
                    )

                    Spacer(Modifier.height(5.dp))

                    Text(
                        analytics.totalStudents.toString(),
                        color = White,
                        fontSize = 36.sp,
                        fontWeight = FontWeight.ExtraBold
                    )

                    Text(
                        "Students included in the analysis",
                        color = Color.White.copy(alpha = 0.8f),
                        fontSize = 11.sp
                    )
                }

                Text(
                    "395",
                    color = Color.White.copy(alpha = 0.15f),
                    fontSize = 45.sp,
                    fontWeight = FontWeight.ExtraBold
                )
            }
        }

        // OVERALL PERFORMANCE

        CardSection {

            Text(
                "Overall Student Performance",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = DarkText
            )

            Text(
                "An overview of academic achievement across the student population.",
                color = MutedText,
                fontSize = 12.sp
            )

            Spacer(Modifier.height(16.dp))

            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {

                PerformanceStat(
                    title = "Average Grade",
                    value = formatNumber(analytics.averageGrade, 1),
                    subtitle = "Out of 20",
                    modifier = Modifier.weight(1f)
                )

                PerformanceStat(
                    title = "Highest Grade",
                    value = formatNumber(analytics.highestGrade, 1),
                    subtitle = "Out of 20",
                    modifier = Modifier.weight(1f)
                )

                PerformanceStat(
                    title = "Lowest Grade",
                    value = formatNumber(analytics.lowestGrade, 1),
                    subtitle = "Out of 20",
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(Modifier.height(18.dp))

            HorizontalDivider(color = BorderColor)

            Spacer(Modifier.height(15.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {

                Text(
                    "Overall pass rate",
                    fontWeight = FontWeight.SemiBold,
                    color = DarkText
                )

                Text(
                    "${formatNumber(analytics.passRate, 1)}%",
                    fontWeight = FontWeight.Bold,
                    color = ForestGreen
                )
            }

            Spacer(Modifier.height(8.dp))

            LinearProgressIndicator(
                progress = {
                    (analytics.passRate / 100.0).toFloat()
                        .coerceIn(0f, 1f)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(9.dp),
                color = ForestGreen,
                trackColor = LightGreen
            )

            Spacer(Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {

                Text(
                    "${analytics.passCount} students passed",
                    fontSize = 12.sp,
                    color = ForestGreen,
                    fontWeight = FontWeight.Medium
                )

                Text(
                    "${analytics.failCount} did not pass",
                    fontSize = 12.sp,
                    color = RedColor,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        // PERFORMANCE DISTRIBUTION

        CardSection {

            Text(
                "Performance Distribution",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = DarkText
            )

            Text(
                "Number of students in each grade category.",
                color = MutedText,
                fontSize = 12.sp
            )

            Spacer(Modifier.height(15.dp))

            PerformanceDistributionRow(
                label = "Excellent",
                description = "16–20 marks",
                count = analytics.excellentCount,
                total = analytics.totalStudents,
                color = ForestGreen
            )

            PerformanceDistributionRow(
                label = "Good",
                description = "12–15.9 marks",
                count = analytics.goodCount,
                total = analytics.totalStudents,
                color = Color(0xFF39945A)
            )

            PerformanceDistributionRow(
                label = "Average",
                description = "10–11.9 marks",
                count = analytics.averageCount,
                total = analytics.totalStudents,
                color = Color(0xFFD69E2E)
            )

            PerformanceDistributionRow(
                label = "Below Average",
                description = "7–9.9 marks",
                count = analytics.belowAverageCount,
                total = analytics.totalStudents,
                color = WarningColor
            )

            PerformanceDistributionRow(
                label = "At Risk",
                description = "Below 7 marks",
                count = analytics.atRiskCount,
                total = analytics.totalStudents,
                color = RedColor
            )
        }

        // ACADEMIC SUMMARY

        CardSection {

            Text(
                "Academic Summary",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = DarkText
            )

            InfoRow(
                "Total students",
                analytics.totalStudents.toString()
            )

            InfoRow(
                "Students who passed",
                analytics.passCount.toString()
            )

            InfoRow(
                "Students who did not pass",
                analytics.failCount.toString()
            )

            InfoRow(
                "Average Mathematics grade",
                "${formatNumber(analytics.averageGrade)} / 20"
            )

            InfoRow(
                "Pass rate",
                "${formatNumber(analytics.passRate, 1)}%"
            )
        }

        // INTERPRETATION

        CardSection {

            Text(
                "Performance Overview",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = DarkText
            )

            Spacer(Modifier.height(8.dp))

            Text(
                when {
                    analytics.passRate >= 80.0 ->
                        "The overall student population is demonstrating strong academic performance. Most students are achieving the expected pass mark."

                    analytics.passRate >= 60.0 ->
                        "The overall performance is encouraging, although some students may benefit from additional academic support and revision."

                    analytics.passRate >= 40.0 ->
                        "A considerable proportion of students are experiencing academic difficulties. Targeted intervention and closer progress monitoring may improve outcomes."

                    else ->
                        "The results indicate significant academic challenges across the student population. Focused support, additional revision and individual student monitoring are recommended."
                },
                color = MutedText,
                fontSize = 14.sp
            )
        }

        Text(
            "Student performance is calculated from the available Mathematics records.",
            color = MutedText,
            fontSize = 11.sp
        )
    }
}

// ------------------------------------------------------------
// PERFORMANCE STAT
// ------------------------------------------------------------

@Composable
fun PerformanceStat(
    title: String,
    value: String,
    subtitle: String,
    modifier: Modifier = Modifier
) {

    Column(
        modifier = modifier
            .background(
                PaleBackground,
                RoundedCornerShape(10.dp)
            )
            .padding(10.dp)
    ) {

        Text(
            value,
            color = ForestGreen,
            fontWeight = FontWeight.Bold,
            fontSize = 21.sp
        )

        Text(
            title,
            color = DarkText,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold
        )

        Text(
            subtitle,
            color = MutedText,
            fontSize = 10.sp
        )
    }
}

// ------------------------------------------------------------
// PERFORMANCE DISTRIBUTION ROW
// ------------------------------------------------------------

@Composable
fun PerformanceDistributionRow(
    label: String,
    description: String,
    count: Int,
    total: Int,
    color: Color
) {

    val percentage = if (total > 0) {
        count.toFloat() / total.toFloat()
    } else {
        0f
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 9.dp)
    ) {

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {

            Column {

                Text(
                    label,
                    color = DarkText,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp
                )

                Text(
                    description,
                    color = MutedText,
                    fontSize = 11.sp
                )
            }

            Column(
                horizontalAlignment = Alignment.End
            ) {

                Text(
                    count.toString(),
                    color = DarkText,
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp
                )

                Text(
                    "${formatNumber(percentage * 100.0, 1)}%",
                    color = MutedText,
                    fontSize = 10.sp
                )
            }
        }

        Spacer(Modifier.height(7.dp))

        LinearProgressIndicator(
            progress = {
                percentage.coerceIn(0f, 1f)
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(7.dp),
            color = color,
            trackColor = LightGreen
        )
    }
}

// ------------------------------------------------------------
// COMMON CARD
// ------------------------------------------------------------

@Composable
fun CardSection(
    content: @Composable ColumnScope.() -> Unit
) {

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = White
        )
    ) {

        Column(
            modifier = Modifier.padding(16.dp),
            content = content
        )
    }
}

// ------------------------------------------------------------
// INFORMATION ROW
// ------------------------------------------------------------

@Composable
fun InfoRow(
    label: String,
    value: String
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 7.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {

        Text(
            label,
            color = MutedText,
            fontSize = 13.sp
        )

        Text(
            value,
            color = DarkText,
            fontSize = 13.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

// ------------------------------------------------------------
// MARKLIST
// ------------------------------------------------------------

@Composable
fun MarklistScreen(records: List<StudentRecord>) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {

        Text(
            "Student Marklist",
            fontSize = 21.sp,
            fontWeight = FontWeight.Bold,
            color = DarkText
        )

        Text(
            "Historical student records",
            color = MutedText,
            fontSize = 12.sp
        )

        Spacer(Modifier.height(12.dp))

        Card(
            modifier = Modifier.fillMaxSize(),
            colors = CardDefaults.cardColors(
                containerColor = White
            ),
            shape = RoundedCornerShape(12.dp)
        ) {

            Column {

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                ) {

                    Row(
                        modifier = Modifier
                            .background(LightGreen)
                            .padding(vertical = 12.dp)
                    ) {

                        TableCell("Student", 105.dp, true)
                        TableCell("G1", 65.dp, true)
                        TableCell("G2", 65.dp, true)
                        TableCell("Study", 75.dp, true)
                        TableCell("Failures", 80.dp, true)
                        TableCell("Absences", 85.dp, true)
                        TableCell("Activities", 90.dp, true)
                        TableCell("Romantic", 85.dp, true)
                        TableCell("G3", 75.dp, true)
                    }

                    Column(
                        modifier = Modifier.verticalScroll(
                            rememberScrollState()
                        )
                    ) {

                        records.forEach { student ->

                            Row(
                                modifier = Modifier.padding(vertical = 10.dp)
                            ) {

                                TableCell(
                                    "Student ${student.id}",
                                    105.dp
                                )

                                TableCell(student.g1.toString(), 65.dp)
                                TableCell(student.g2.toString(), 65.dp)
                                TableCell(student.studyTime.toString(), 75.dp)
                                TableCell(student.failures.toString(), 80.dp)
                                TableCell(student.absences.toString(), 85.dp)

                                TableCell(
                                    if (student.activities) "Yes" else "No",
                                    90.dp
                                )

                                TableCell(
                                    if (student.romantic) "Yes" else "No",
                                    85.dp
                                )

                                TableCell(
                                    formatNumber(student.g3, 1),
                                    75.dp
                                )
                            }

                            HorizontalDivider(
                                color = BorderColor,
                                thickness = 0.5.dp
                            )
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// TABLE CELL
// ------------------------------------------------------------

@Composable
fun TableCell(
    value: String,
    width: Dp,
    header: Boolean = false
) {

    Text(
        text = value,
        modifier = Modifier
            .width(width)
            .padding(horizontal = 8.dp),
        fontSize = if (header) 12.sp else 11.sp,
        fontWeight = if (header) {
            FontWeight.Bold
        } else {
            FontWeight.Normal
        },
        color = if (header) ForestGreen else DarkText
    )
}

// ------------------------------------------------------------
// PREDICTION SCREEN
// ------------------------------------------------------------

@Composable
fun PredictionScreen(model: RandomForestRegressor) {

    var g1 by remember { mutableStateOf("12") }
    var g2 by remember { mutableStateOf("14") }
    var studyTime by remember { mutableStateOf("2") }
    var failures by remember { mutableStateOf("0") }
    var absences by remember { mutableStateOf("3") }

    var activities by remember { mutableStateOf(true) }
    var romantic by remember { mutableStateOf(false) }

    var result by remember { mutableStateOf<Double?>(null) }
    var error by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(13.dp)
    ) {

        Text(
            "Predict a student's grade",
            fontSize = 21.sp,
            fontWeight = FontWeight.Bold,
            color = DarkText
        )

        Text(
            "Enter the student's academic information to estimate their final Mathematics grade.",
            color = MutedText,
            fontSize = 13.sp
        )

        CardSection {

            NumberField("First-period grade (G1)", g1) {
                g1 = it
                result = null
            }

            NumberField("Second-period grade (G2)", g2) {
                g2 = it
                result = null
            }

            NumberField("Study time (1–4)", studyTime) {
                studyTime = it
                result = null
            }

            NumberField("Previous failures", failures) {
                failures = it
                result = null
            }

            NumberField("Absences", absences) {
                absences = it
                result = null
            }

            Spacer(Modifier.height(8.dp))

            Text(
                "Extracurricular activities",
                fontWeight = FontWeight.SemiBold,
                color = DarkText
            )

            SwitchRow(
                "Participates in activities",
                activities
            ) {
                activities = it
                result = null
            }

            Spacer(Modifier.height(8.dp))

            Text(
                "Additional information",
                fontWeight = FontWeight.SemiBold,
                color = DarkText
            )

            SwitchRow(
                "In a romantic relationship",
                romantic
            ) {
                romantic = it
                result = null
            }

            Spacer(Modifier.height(15.dp))

            Button(
                onClick = {

                    val first = g1.toDoubleOrNull()
                    val second = g2.toDoubleOrNull()
                    val study = studyTime.toDoubleOrNull()
                    val fail = failures.toDoubleOrNull()
                    val absent = absences.toDoubleOrNull()

                    error = null
                    result = null

                    if (
                        first == null ||
                        second == null ||
                        study == null ||
                        fail == null ||
                        absent == null
                    ) {

                        error = "Please enter valid numeric values."

                    } else if (
                        first !in 0.0..20.0 ||
                        second !in 0.0..20.0 ||
                        study !in 1.0..4.0 ||
                        study % 1.0 != 0.0 ||
                        fail < 0.0 ||
                        fail % 1.0 != 0.0 ||
                        absent < 0.0 ||
                        absent % 1.0 != 0.0
                    ) {

                        error = "Check the permitted ranges of your entries."

                    } else {

                        val input = StudentInput(
                            g1 = first,
                            g2 = second,
                            studyTime = study,
                            failures = fail,
                            absences = absent,
                            activities = activities,
                            romantic = romantic
                        )

                        result = model.predict(
                            FeatureEncoder.encode(input)
                        ).coerceIn(0.0, 20.0)
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = ForestGreen
                ),
                shape = RoundedCornerShape(10.dp)
            ) {

                Text(
                    "Predict Mathematics Grade",
                    modifier = Modifier.padding(5.dp)
                )
            }

            error?.let {

                Text(
                    it,
                    color = Color.Red,
                    fontSize = 13.sp
                )
            }
        }

        result?.let { grade ->

            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = LightGreen
                ),
                shape = RoundedCornerShape(14.dp)
            ) {

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(22.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {

                    Text(
                        "Predicted final grade",
                        color = ForestGreen,
                        fontWeight = FontWeight.SemiBold
                    )

                    Text(
                        formatNumber(grade),
                        fontSize = 45.sp,
                        color = ForestGreen,
                        fontWeight = FontWeight.ExtraBold
                    )

                    Text(
                        "Out of 20",
                        color = MutedText
                    )

                    Spacer(Modifier.height(12.dp))

                    Text(
                        when {
                            grade >= 16 ->
                                "Strong predicted performance"

                            grade >= 12 ->
                                "Good predicted performance"

                            grade >= 10 ->
                                "Moderate predicted performance"

                            grade >= 7 ->
                                "Below-target predicted performance"

                            else ->
                                "At-risk predicted performance"
                        },
                        color = DarkText,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(Modifier.height(8.dp))

                    Text(
                        "This is a statistical estimate, not a guaranteed result.",
                        color = MutedText,
                        fontSize = 11.sp
                    )
                }
            }
        }
    }
}

// ------------------------------------------------------------
// NUMBER FIELD
// ------------------------------------------------------------

@Composable
fun NumberField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {

    Column(
        modifier = Modifier.padding(bottom = 10.dp)
    ) {

        Text(
            label,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            color = DarkText
        )

        Spacer(Modifier.height(5.dp))

        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            keyboardOptions = KeyboardOptions(
                keyboardType = KeyboardType.Number
            ),
            shape = RoundedCornerShape(9.dp)
        )
    }
}

// ------------------------------------------------------------
// SWITCH ROW
// ------------------------------------------------------------

@Composable
fun SwitchRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {

        Text(
            label,
            fontSize = 13.sp,
            color = MutedText
        )

        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = White,
                checkedTrackColor = ForestGreen
            )
        )
    }
}

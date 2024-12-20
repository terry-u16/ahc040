//! ビームサーチライブラリ
//! [rhooさんの記事](https://qiita.com/rhoo/items/2f647e32f6ff2c6ee056)を大いに参考にさせて頂きました。
//! ありがとうございます……。
//!
//! # Usage
//!
//! ```
//! use cp_lib_rs::beam;
//!
//! // 状態のうち、差分計算を行わない部分
//! #[derive(Clone, Default)]
//! struct SmallState;
//!
//! impl beam::SmallState for SmallState {
//!     type Score = i64;
//!     type Hash = u64;
//!     type LargeState = LargeState;
//!     type Action = usize;
//!
//!     fn raw_score(&self) -> <Self as beam::SmallState>::Score { todo!() }
//!     fn hash(&self) -> <Self as beam::SmallState>::Hash { todo!() }
//!     fn apply(&self, state: &mut <Self as beam::SmallState>::LargeState) { todo!() }
//!     fn rollback(&self, state: &mut <Self as beam::SmallState>::LargeState) { todo!() }
//!     fn action(&self) -> <Self as beam::SmallState>::Action { todo!() }
//! }
//!
//! // 状態のうち、差分計算を行う部分
//! struct LargeState;
//!
//! // 次の行動を生成する構造体
//! struct ActionGenerator;
//!
//! impl beam::ActGen<SmallState> for ActionGenerator {
//!     fn generate(&self, small_state: &SmallState, large_state: &<SmallState as beam::SmallState>::LargeState, actions: &mut Vec<SmallState>) {
//!         // 状態を生成してactionsに追加する
//!         for _ in 0..0 {
//!            actions.push(SmallState);
//!         }
//!     }
//! }
//!
//! fn beam() -> Vec<usize> {
//!     let large_state = LargeState;
//!     let small_state = SmallState;
//!     let action_generator = ActionGenerator;
//!     let mut beam = beam::BeamSearch::new(large_state, small_state, action_generator);
//!
//!     let deduplicator = beam::NoOpDeduplicator;
//!     let beam_width = beam::FixedBeamWidthSuggester::new(100);
//!     let (actions, score) = beam.run(2500, beam_width, deduplicator);
//!
//!     actions
//! }
//!
//! ```
//!
use std::{
    cmp::Reverse,
    fmt::Display,
    hash::Hash,
    ops::{Index, IndexMut},
    time::Instant,
};

use rustc_hash::FxHashSet;

/// コピー可能な小さい状態を表すトレイト
pub trait SmallState {
    type Score: Ord + Display;
    type Hash: Hash + Eq;
    type LargeState;
    type Action;

    /// ビームサーチ用スコア（大きいほど良い）
    /// デフォルトでは生スコアをそのまま返す
    fn beam_score(&self) -> Self::Score {
        self.raw_score()
    }

    // 生スコア
    fn raw_score(&self) -> Self::Score;

    /// ハッシュ値
    fn hash(&self) -> Self::Hash;

    /// stateにこの差分を作用させる
    fn apply(&self, state: &mut Self::LargeState);

    /// stateに作用させたこの差分をロールバックする
    fn rollback(&self, state: &mut Self::LargeState);

    /// 実行した行動を返す
    fn action(&self) -> Self::Action;
}

/// 現在のstateからの遷移先を列挙するトレイト
pub trait ActGen<S: SmallState> {
    /// 現在のstateからの遷移先をnext_satesに格納する
    fn generate(&self, small_state: &S, large_state: &S::LargeState, next_states: &mut Vec<S>);
}

/// ビームの次の遷移候補
#[derive(Clone)]
struct Cancidate<S: SmallState + Clone> {
    /// 実行後のsmall_state
    small_state: S,
    /// 親となるノードのインデックス
    parent: NodeIndex,
}

impl<S: SmallState + Clone> Cancidate<S> {
    fn new(small_state: S, parent: NodeIndex) -> Self {
        Self {
            small_state,
            parent,
        }
    }

    fn to_node(
        self,
        child: NodeIndex,
        left_sibling: NodeIndex,
        right_sibling: NodeIndex,
    ) -> Node<S> {
        Node {
            small_state: self.small_state,
            parent: self.parent,
            child,
            left_sibling,
            right_sibling,
        }
    }
}

/// 重複除去を行うトレイト
pub trait Deduplicator<S: SmallState> {
    /// 重複除去に使った情報をクリアし、次の重複除去の準備をする
    fn clear(&mut self);

    /// 重複チェックを行い、残すべきならtrue、重複していればfalseを返す
    fn filter(&mut self, state: &S) -> bool;
}

/// 重複除去を行わず素通しするDeduplicator
pub struct NoOpDeduplicator;

impl<S: SmallState> Deduplicator<S> for NoOpDeduplicator {
    fn clear(&mut self) {
        // do nothing
    }

    fn filter(&mut self, _state: &S) -> bool {
        // 常に素通しする
        true
    }
}

/// 同じハッシュ値を持つ状態を1つだけに制限するDeduplicator
pub struct HashSingleDeduplicator<S: SmallState> {
    set: FxHashSet<S::Hash>,
}

impl<S: SmallState> HashSingleDeduplicator<S> {
    pub fn new() -> Self {
        Self {
            set: FxHashSet::default(),
        }
    }
}

impl<S: SmallState> Deduplicator<S> for HashSingleDeduplicator<S> {
    fn clear(&mut self) {
        self.set.clear();
    }

    fn filter(&mut self, state: &S) -> bool {
        // ハッシュが重複していなければ通す
        self.set.insert(state.hash())
    }
}

/// ビーム幅を提案するトレイト
pub trait BeamWidthSuggester {
    // 現在のターン数を受け取り、ビーム幅を提案する
    fn suggest(&mut self) -> usize;
}

/// 常に固定のビーム幅を返すBeamWidthSuggester
pub struct FixedBeamWidthSuggester {
    width: usize,
}

impl FixedBeamWidthSuggester {
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

impl BeamWidthSuggester for FixedBeamWidthSuggester {
    fn suggest(&mut self) -> usize {
        self.width
    }
}

/// ベイズ推定+カルマンフィルタにより適切なビーム幅を計算するBeamWidthSuggester。
/// 1ターンあたりの実行時間が正規分布に従うと仮定し、+3σ分の余裕を持ってビーム幅を決める。
///
/// ## モデル
///
/// カルマンフィルタを適用するにあたって、以下のモデルを考える。
///
/// - `i` ターン目のビーム幅1あたりの所要時間の平均値 `t_i` が正規分布 `N(μ_i, σ_i^2)` に従うと仮定する。
///   - 各ターンに観測される所要時間が `N(μ_i, σ_i^2)` に従うのではなく、所要時間の**平均値**が `N(μ_i, σ_i^2)` に従うとしている点に注意。
///     - すなわち `μ_i` は所要時間の平均値の平均値であり、所要時間の平均値が `μ_i` を中心とした確率分布を形成しているものとしている。ややこしい。
///   - この `μ_i` , `σ_i^2` をベイズ推定によって求めたい。
/// - 所要時間 `t_i` は `t_{i+1}=t_i+N(0, α^2)` により更新されるものとする。
///   - `N(0, α^2)` は標準偏差 `α` のノイズを意味する。お気持ちとしては「実行時間がターン経過に伴ってちょっとずつ変わっていくことがあるよ」という感じ。
///   - `α` は既知の定数とし、適当に決める。
///   - 本来は問題に合わせたちゃんとした更新式にすべき（ターン経過に伴って線形に増加するなど）なのだが、事前情報がないため大胆に仮定する。
/// - 所要時間の観測値 `τ_i` は `τ_i=t_i+N(0, β^2)` により得られるものとする。
///   - `β` は既知の定数とし、適当に決める。
///   - 本来この `β` も推定できると嬉しいのだが、取扱いが煩雑になるためこちらも大胆に仮定する。
///
/// ## モデルの初期化
///
/// - `μ_0` は実行時間制限を `T` 、標準ビーム幅を `W` 、実行ターン数を `M` として、 `μ_0=T/WM` などとすればよい。
/// - `σ_0` は適当に `σ_0=0.1μ_0` とする。ここは標準ビーム幅にどのくらい自信があるかによる。
/// - `α` は適当に `α=0.01μ_0` とする。定数は本当に勘。多分問題に合わせてちゃんと考えた方が良い。
/// - `β` は `σ_0=0.05μ_0` とする。適当なベンチマーク問題で標準偏差を取ったらそのくらいだったため。
///
/// ## モデルの更新
///
/// 以下のように更新をかけていく。
///
/// 1. `t_0=N(μ_0, σ_0^2)` と初期化する。
/// 2. `t_1=t_0+N(0, α^2)` とし、事前分布 `t_1=N(μ_1, σ_1^2)=N(μ_0, σ_0^2+α^2)` を得る。ここはベイズ更新ではなく単純な正規分布の合成でよい。
/// 3. `τ_1` が観測されるので、ベイズ更新して事後分布 `N(μ_1', σ_1^2')` を得る。
/// 4. 同様に `t_2=N(μ_2, σ_2^2)` を得る。
/// 5. `τ_2` を用いてベイズ更新。以下同様。
///
/// ## 適切なビーム幅の推定
///
/// - 余裕を持って、99.8%程度の確率（+3σ）で実行時間制限に収まるようなビーム幅にしたい。
/// - ここで、 `t_i=t_{i+1}=･･･=t_M=N(μ_i, σ_i^2)` と大胆仮定する。
///   - `α` によって `t_i` がどんどん変わってしまうと考えるのは保守的すぎるため。
/// - すると残りターン数 `M_i=M-i` として、 `Στ_i=N(M_i*μ_i, M_i*σ_i^2)` となる。
/// - したがって、残り時間を `T_i` として `W(M_i*μ_i+3(σ_i√M_i))≦T_i` となる最大の `W` を求めればよく、 `W=floor(T_i/(M_i*μ_i+3(σ_i√M_i)))` となる。
/// - 最後に、念のため適当な `W_min` , `W_max` でclampしておく。
pub struct BayesianBeamWidthSuggester {
    /// ビーム幅1あたりの所要時間の平均値の平均値μ_i（逐次更新される）
    mean_sec: f64,
    /// ビーム幅1あたりの所要時間の平均値の分散σ_i^2（逐次更新される）
    variance_sec: f64,
    /// 1ターンごとに状態に作用するノイズの大きさを表す分散α^2（定数）
    variance_state_sec: f64,
    /// 観測時に乗るノイズの大きさを表す分散β^2（定数）
    variance_observe_sec: f64,
    /// 問題の実行時間制限T
    time_limit_sec: f64,
    /// 現在のターン数i
    current_turn: usize,
    /// 最大ターン数M
    max_turn: usize,
    /// ウォームアップターン数（最初のXターン分の情報は採用せずに捨てる）
    warmup_turn: usize,
    /// 最小ビーム幅W_min
    min_beam_width: usize,
    /// 最大ビーム幅W_max
    max_beam_width: usize,
    /// 現在のビーム幅W_i
    current_beam_width: usize,
    /// ログの出力インターバル（0にするとログを出力しなくなる）
    verbose_interval: usize,
    /// ビーム開始時刻
    start_time: Instant,
    /// 前回の計測時刻
    last_time: Instant,
}

impl BayesianBeamWidthSuggester {
    pub fn new(
        max_turn: usize,
        warmup_turn: usize,
        time_limit_sec: f64,
        standard_beam_width: usize,
        min_beam_width: usize,
        max_beam_width: usize,
        verbose_interval: usize,
    ) -> Self {
        assert!(
            max_turn * standard_beam_width > 0,
            "ターン数とビーム幅設定が不正です。"
        );
        assert!(
            min_beam_width > 0,
            "最小のビーム幅は正の値でなければなりません。"
        );
        assert!(
            min_beam_width <= max_beam_width,
            "最大のビーム幅は最小のビーム幅以上でなければなりません。"
        );

        let mean_sec = time_limit_sec / (max_turn * standard_beam_width) as f64;

        // 雑にσ=10%ズレると仮定
        let stddev_sec = 0.1 * mean_sec;
        let variance_sec = stddev_sec * stddev_sec;
        let stddev_state_sec = 0.01 * mean_sec;
        let variance_state_sec = stddev_state_sec * stddev_state_sec;
        let stddev_observe_sec = 0.05 * mean_sec;
        let variance_observe_sec = stddev_observe_sec * stddev_observe_sec;

        Self {
            mean_sec,
            variance_sec,
            time_limit_sec,
            variance_state_sec,
            variance_observe_sec,
            current_turn: 0,
            min_beam_width,
            max_beam_width,
            verbose_interval,
            max_turn,
            warmup_turn,
            current_beam_width: 0,
            start_time: Instant::now(),
            last_time: Instant::now(),
        }
    }

    fn update_state(&mut self) {
        // N(0, α^2)のノイズが乗る
        self.variance_sec += self.variance_state_sec;
    }

    fn update_distribution(&mut self, duration_sec: f64) {
        let old_mean = self.mean_sec;
        let old_variance = self.variance_sec;
        let noise_variance = self.variance_observe_sec;

        self.mean_sec = (old_mean * noise_variance + old_variance * duration_sec)
            / (noise_variance + old_variance);
        self.variance_sec = old_variance * noise_variance / (old_variance + noise_variance);
    }

    fn calc_safe_beam_width(&self) -> usize {
        let remaining_turn = (self.max_turn - self.current_turn) as f64;
        let elapsed_time = (Instant::now() - self.start_time).as_secs_f64();
        let remaining_time = self.time_limit_sec - elapsed_time;

        // 平均値の分散σ^2と観測ノイズβ^2が乗ってくると考える
        let variance_total = self.variance_sec + self.variance_observe_sec;

        // N(ξ, η^2)からのサンプリングをK回繰り返すとN(Kξ, Kη^2)となる（はず）
        let mean = remaining_turn * self.mean_sec;
        let variance = remaining_turn * variance_total;
        let stddev = variance.sqrt();

        // 3σの余裕を持たせる
        const SIGMA_COEF: f64 = 3.0;
        let needed_time_per_width = mean + SIGMA_COEF * stddev;
        let beam_width = ((remaining_time / needed_time_per_width) as usize)
            .max(self.min_beam_width)
            .min(self.max_beam_width);

        if self.verbose_interval != 0 && self.current_turn % self.verbose_interval == 0 {
            let stddev_per_run = (self.max_turn as f64 * variance_total).sqrt();
            let stddev_per_turn = variance_total.sqrt();

            eprintln!(
                "turn: {:4}, beam width: {:4}, pase: {:.3}±{:.3}ms/run, iter time: {:.3}±{:.3}ms",
                self.current_turn,
                beam_width,
                self.mean_sec * (beam_width * self.max_turn) as f64 * 1e3,
                stddev_per_run * beam_width as f64 * 1e3,
                self.mean_sec * beam_width as f64 * 1e3,
                stddev_per_turn * beam_width as f64 * 1e3
            );
        }

        beam_width
    }
}

impl BeamWidthSuggester for BayesianBeamWidthSuggester {
    fn suggest(&mut self) -> usize {
        assert!(
            self.current_turn < self.max_turn,
            "規定ターン終了後にsuggest()が呼び出されました。"
        );

        if self.current_turn >= self.warmup_turn {
            let elapsed = (Instant::now() - self.last_time).as_secs_f64();
            let elapsed_per_beam = elapsed / self.current_beam_width as f64;
            self.update_state();
            self.update_distribution(elapsed_per_beam);
        }

        self.last_time = Instant::now();
        let beam_width = self.calc_safe_beam_width();
        self.current_beam_width = beam_width;
        self.current_turn += 1;
        beam_width
    }
}

/// ビームサーチ木のノード
#[derive(Debug, Default, Clone)]
struct Node<S: SmallState> {
    /// 実行後のsmall_state
    small_state: S,
    /// （N分木と考えたときの）親ノード
    parent: NodeIndex,
    /// （二重連鎖木と考えたときの）子ノード
    child: NodeIndex,
    /// （二重連鎖木と考えたときの）左の兄弟ノード
    left_sibling: NodeIndex,
    /// （二重連鎖木と考えたときの）右の兄弟ノード
    right_sibling: NodeIndex,
}

impl<S: SmallState> Node<S> {
    fn new(
        small_state: S,
        parent: NodeIndex,
        child: NodeIndex,
        left_sibling: NodeIndex,
        right_sibling: NodeIndex,
    ) -> Self {
        Self {
            small_state,
            parent,
            child,
            left_sibling,
            right_sibling,
        }
    }
}

/// NodeVec用のindex
/// 型安全性と、indexの内部的な型(u32 or u16)の変更を容易にすることが目的
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NodeIndex(usize);

impl NodeIndex {
    /// 何も指していないことを表す定数
    const NULL: NodeIndex = NodeIndex(!0);
}

impl Default for NodeIndex {
    fn default() -> Self {
        Self::NULL
    }
}

impl From<usize> for NodeIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl Into<usize> for NodeIndex {
    fn into(self) -> usize {
        self.0 as usize
    }
}

/// Nodeのコレクション
#[derive(Debug)]
struct NodeVec<S: SmallState> {
    nodes: Vec<Node<S>>,
    free_indices: Vec<usize>,
}

impl<S: SmallState + Default + Clone> NodeVec<S> {
    fn new(capacity: usize) -> Self {
        Self {
            nodes: vec![Default::default(); capacity],
            free_indices: (0..capacity).rev().collect(),
        }
    }

    fn push(&mut self, node: Node<S>) -> NodeIndex {
        let index = self
            .free_indices
            .pop()
            .expect("ノードプールの容量制限に達しました。");

        self.nodes[index] = node;

        NodeIndex::from(index)
    }

    fn delete(&mut self, index: NodeIndex) {
        self.free_indices.push(index.into());
    }
}

impl<S: SmallState> Index<NodeIndex> for NodeVec<S> {
    type Output = Node<S>;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        let index: usize = index.into();
        self.nodes.index(index)
    }
}

impl<S: SmallState> IndexMut<NodeIndex> for NodeVec<S> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        let index: usize = index.into();
        self.nodes.index_mut(index)
    }
}

/// 保持する最大ノード数
const MAX_NODES: usize = 131072;

#[derive(Debug)]
pub struct BeamSearch<S: SmallState, G: ActGen<S>> {
    act_gen: G,
    nodes: NodeVec<S>,
    current_index: NodeIndex,
    leaves: Vec<NodeIndex>,
    next_leaves: Vec<NodeIndex>,
    action_buffer: Vec<S>,
}

impl<S: SmallState + Default + Clone, G: ActGen<S>> BeamSearch<S, G> {
    /// ビーム木を指定された容量で初期化する
    pub fn new(act_gen: G) -> Self {
        let nodes = NodeVec::new(MAX_NODES);

        Self {
            act_gen,
            nodes,
            current_index: NodeIndex(0),
            leaves: vec![],
            next_leaves: vec![],
            action_buffer: vec![],
        }
    }

    fn init(&mut self, small_state: S) {
        // 参照局所性が高くなるようにお祈りしながらソートする
        self.leaves.sort_unstable();

        while let Some(index) = self.leaves.pop() {
            self.nodes.delete(index);
        }

        self.nodes.push(Node::new(
            small_state,
            NodeIndex::NULL,
            NodeIndex::NULL,
            NodeIndex::NULL,
            NodeIndex::NULL,
        ));

        self.current_index = NodeIndex(0);
        self.leaves.push(NodeIndex(0));
        self.next_leaves.clear();
        self.action_buffer.clear();
    }

    pub fn run<W: BeamWidthSuggester, P: Deduplicator<S>>(
        &mut self,
        mut large_state: S::LargeState,
        small_state: S,
        max_turn: usize,
        mut beam_width_suggester: W,
        mut deduplicator: P,
    ) -> (Vec<S::Action>, S::Score) {
        // 初期化
        self.init(small_state);

        let mut candidates = vec![];
        let mut indices = vec![];

        for turn in 0..max_turn {
            let beam_width = beam_width_suggester.suggest();
            candidates.clear();
            self.dfs(&mut candidates, true, &mut large_state);

            if turn + 1 == max_turn {
                break;
            }

            assert_ne!(
                candidates.len(),
                0,
                "次の状態の候補が見つかりませんでした。"
            );

            // 重複除去を行ったのち、次の遷移先を確定させる
            indices.clear();
            indices.extend(0..candidates.len());
            glidesort::sort_by_key(&mut indices, |&i| {
                Reverse(candidates[i].small_state.beam_score())
            });

            deduplicator.clear();
            self.update_tree(
                indices
                    .iter()
                    .map(|&i| candidates[i].clone())
                    .filter(|c| deduplicator.filter(&c.small_state))
                    .take(beam_width),
            );
        }

        let Cancidate {
            small_state,
            parent,
            ..
        } = candidates
            .into_iter()
            .max_by_key(|c| c.small_state.raw_score())
            .expect("最終状態となる候補が見つかりませんでした。");

        // 操作列の復元
        let mut actions = self.restore_actions(parent);
        actions.push(small_state.action());
        (actions, small_state.raw_score())
    }

    /// ノードを追加する
    fn add_node(&mut self, candidate: Cancidate<S>) {
        let parent = candidate.parent;
        let node_index =
            self.nodes
                .push(candidate.to_node(NodeIndex::NULL, NodeIndex::NULL, NodeIndex::NULL));

        // 親の子、すなわち一番左にいる兄弟ノード
        let sibling = self.nodes[parent].child;

        // 既に兄弟がいる場合、その左側に入る
        if sibling != NodeIndex::NULL {
            self.nodes[sibling].left_sibling = node_index;
        }

        // 兄弟を1マス右に押し出して、自分が一番左に入る
        self.next_leaves.push(node_index);
        self.nodes[parent].child = node_index;
        self.nodes[node_index].right_sibling = sibling;
    }

    /// 指定されたインデックスのノードを削除する
    /// 必要に応じてビーム木の辺を繋ぎ直す
    fn remove_node(&mut self, mut index: NodeIndex) {
        loop {
            let Node {
                left_sibling,
                right_sibling,
                parent,
                ..
            } = self.nodes[index];
            self.nodes.delete(index);

            // 親は生きているはず
            assert_ne!(parent, NodeIndex::NULL, "rootノードを消そうとしています。");

            // もう兄弟がいなければ親へ
            if left_sibling == NodeIndex::NULL && right_sibling == NodeIndex::NULL {
                index = parent;
                continue;
            }

            // 左右の連結リストを繋ぎ直す
            if left_sibling != NodeIndex::NULL {
                self.nodes[left_sibling].right_sibling = right_sibling;
            } else {
                self.nodes[parent].child = right_sibling;
            }

            if right_sibling != NodeIndex::NULL {
                self.nodes[right_sibling].left_sibling = left_sibling;
            }

            return;
        }
    }

    /// DFSでビームサーチ木を走査し、次の状態の一覧をcandidatesに詰める
    /// ビームサーチ木が一本道の場合は戻る必要がないため、is_single_pathで管理
    fn dfs(
        &mut self,
        candidates: &mut Vec<Cancidate<S>>,
        is_single_path: bool,
        state: &mut S::LargeState,
    ) {
        // 葉ノードであれば次の遷移を行う
        if self.nodes[self.current_index].child == NodeIndex::NULL {
            self.act_gen.generate(
                &self.nodes[self.current_index].small_state,
                &state,
                &mut self.action_buffer,
            );

            while let Some(state) = self.action_buffer.pop() {
                candidates.push(Cancidate::new(state, self.current_index));
            }

            return;
        }

        let current_index = self.current_index;
        let mut child_index = self.nodes[current_index].child;
        let next_is_single_path =
            is_single_path & (self.nodes[child_index].right_sibling == NodeIndex::NULL);

        // デバッグ用
        //let prev_state = self.state.clone();

        // 兄弟ノードを全て走査する
        loop {
            self.current_index = child_index;
            self.nodes[child_index].small_state.apply(state);
            self.dfs(candidates, next_is_single_path, state);

            if !next_is_single_path {
                self.nodes[child_index].small_state.rollback(state);

                // デバッグ用
                //assert!(prev_state == self.state);
            }

            child_index = self.nodes[child_index].right_sibling;

            if child_index == NodeIndex::NULL {
                break;
            }
        }

        if !next_is_single_path {
            self.current_index = current_index;
        }
    }

    /// 木を更新する
    /// 具体的には以下の処理を行う
    ///
    /// - 新しいcandidatesを葉に追加する
    /// - 1ターン前のノードであって葉のノード（今後参照されないノード）を削除する
    fn update_tree(&mut self, candidates: impl Iterator<Item = Cancidate<S>>) {
        self.next_leaves.clear();
        for candidate in candidates {
            self.add_node(candidate);
        }

        for i in 0..self.leaves.len() {
            let node_index = self.leaves[i];

            if self.nodes[node_index].child == NodeIndex::NULL {
                self.remove_node(node_index);
            }
        }

        std::mem::swap(&mut self.leaves, &mut self.next_leaves);
    }

    /// 操作列を復元する
    fn restore_actions(&self, mut index: NodeIndex) -> Vec<S::Action> {
        let mut actions = vec![];

        while self.nodes[index].parent != NodeIndex::NULL {
            actions.push(self.nodes[index].small_state.action());
            index = self.nodes[index].parent;
        }

        actions.reverse();
        actions
    }
}

#[cfg(test)]
mod test {
    //! TSPをビームサーチで解くテスト
    use super::{ActGen, BeamSearch, FixedBeamWidthSuggester, NoOpDeduplicator};

    #[derive(Debug, Clone)]
    struct Input {
        n: usize,
        distances: Vec<Vec<i32>>,
    }

    impl Input {
        fn gen_testcase() -> Self {
            let n = 4;
            let distances = vec![
                vec![0, 2, 3, 10],
                vec![2, 0, 1, 3],
                vec![3, 1, 0, 2],
                vec![10, 3, 2, 0],
            ];

            Self { n, distances }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SmallState {
        distance: i32,
        position: usize,
        visited_count: usize,
    }

    impl SmallState {
        fn new(distance: i32, position: usize, visited_count: usize) -> Self {
            Self {
                distance,
                position,
                visited_count,
            }
        }
    }

    impl Default for SmallState {
        fn default() -> Self {
            Self {
                distance: 0,
                position: 0,
                visited_count: 1,
            }
        }
    }

    impl super::SmallState for SmallState {
        type Score = i32;
        type Hash = u64;
        type LargeState = LargeState;
        type Action = usize;

        fn raw_score(&self) -> Self::Score {
            self.distance
        }

        fn beam_score(&self) -> Self::Score {
            // 大きいほど良いとする
            -self.distance
        }

        fn hash(&self) -> Self::Hash {
            // 適当に0を返す
            0
        }

        fn apply(&self, state: &mut Self::LargeState) {
            // 現在地を訪問済みにする
            state.visited[self.position] = true;
        }

        fn rollback(&self, state: &mut Self::LargeState) {
            // 現在地を未訪問にする
            state.visited[self.position] = false;
        }

        fn action(&self) -> Self::Action {
            self.position
        }
    }

    #[derive(Debug, Clone)]
    struct LargeState {
        visited: Vec<bool>,
    }

    impl LargeState {
        fn new(n: usize) -> Self {
            let mut visited = vec![false; n];
            visited[0] = true;
            Self { visited }
        }
    }

    #[derive(Debug, Clone)]
    struct ActionGenerator<'a> {
        input: &'a Input,
    }

    impl<'a> ActionGenerator<'a> {
        fn new(input: &'a Input) -> Self {
            Self { input }
        }
    }

    impl<'a> ActGen<SmallState> for ActionGenerator<'a> {
        fn generate(
            &self,
            small_state: &SmallState,
            large_state: &LargeState,
            next_states: &mut Vec<SmallState>,
        ) {
            if small_state.visited_count == self.input.n {
                // 頂点0に戻るしかない
                let next_pos = 0;
                let next_dist =
                    small_state.distance + self.input.distances[small_state.position][0];
                let next_visited_count = small_state.visited_count + 1;
                let next_state = SmallState::new(next_dist, next_pos, next_visited_count);
                next_states.push(next_state);
                return;
            }

            // 未訪問の頂点に移動
            for next_pos in 0..self.input.n {
                if large_state.visited[next_pos] {
                    continue;
                }

                let next_dist =
                    small_state.distance + self.input.distances[small_state.position][next_pos];
                let next_visited_count = small_state.visited_count + 1;
                let next_state = SmallState::new(next_dist, next_pos, next_visited_count);
                next_states.push(next_state);
            }
        }
    }

    #[test]
    fn beam_tsp_test() {
        let input = Input::gen_testcase();
        let small_state = SmallState::default();
        let large_state = LargeState::new(input.n);
        let action_generator = ActionGenerator::new(&input);
        let mut beam = BeamSearch::new(action_generator);

        // hashを適当に全て0としているため、重複除去は行わない
        let deduplicator = NoOpDeduplicator;
        let beam_width = FixedBeamWidthSuggester::new(10);

        let (actions, score) =
            beam.run(large_state, small_state, input.n, beam_width, deduplicator);

        eprintln!("score: {}", score);
        eprintln!("actions: {:?}", actions);
        assert_eq!(score, 10);
        assert!(actions == vec![1, 3, 2, 0] || actions == vec![2, 3, 1, 0]);
    }
}

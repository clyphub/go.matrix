// Copyright 2009 The GoMatrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"testing"
)

func TestAdd_Sparse(t *testing.T) {
	A := NormalsSparse(3, 3, 9)
	B := NormalsSparse(3, 3, 9)
	C1, _ := A.Plus(B)
	C2, _ := A.PlusSparse(B)
	if !ApproxEquals(C1, Sum(A, B), ε) {
		t.Fail()
	}
	if !ApproxEquals(C2, Sum(A, B), ε) {
		t.Fail()
	}
}

func TestSubtract_Sparse(t *testing.T) {
	A := NormalsSparse(3, 3, 9)
	B := NormalsSparse(3, 3, 9)
	C1, _ := A.Minus(B)
	C2, _ := A.MinusSparse(B)
	if !ApproxEquals(C1, Difference(A, B), ε) {
		t.Fail()
	}
	if !ApproxEquals(C2, Difference(A, B), ε) {
		t.Fail()
	}
}

func TestTimes_Sparse(t *testing.T) {
	A := Normals(3, 3).SparseMatrix()
	B := Normals(3, 3).SparseMatrix()
	C1, _ := A.Times(B)
	C2, _ := A.TimesSparse(B)
	if !ApproxEquals(C1, Product(A, B), ε) {
		t.Fail()
	}
	if !ApproxEquals(C2, Product(A, B), ε) {
		t.Fail()
	}
}

func TestElementMult_Sparse(t *testing.T) {
	A := Normals(3, 3).SparseMatrix()
	B := Normals(3, 3).SparseMatrix()
	C1, _ := A.ElementMult(B)
	C2, _ := A.ElementMultSparse(B)
	D, _ := A.DenseMatrix().ElementMult(B)
	if !Equals(D, C1) {
		t.Fail()
	}
	if !Equals(D, C2) {
		t.Fail()
	}
}

func TestGetMatrix_Sparse(t *testing.T) {
	numRows := 40
	numColumns := 40
	A := ZerosSparse(numRows, numColumns)
	for i := 0; i < numRows; i++ {
		for j := 0; j < numColumns; j++ {
			A.Set(i, j, float64(100*i+j))
		}
	}
	rowsStart := numRows / 4
	rowsEnd := 2 * numRows / 4
	colsStart := 2 * numColumns / 4
	colsEnd := 3 * numColumns / 4
	B := A.GetMatrix(rowsStart, colsStart, rowsEnd, colsEnd)

	if B.Rows() != numRows/4 {
		t.Log(fmt.Sprintf("wrong number of rows, expected %d, got %d", numRows/4, B.Rows()))
		t.Fail()
	}
	if B.Cols() != numColumns/4 {
		t.Log(fmt.Sprintf("wrong number of columns, expected %d, got %d", numColumns/4, B.Cols()))
		t.Fail()
	}
	for i := 0; i < numRows/4; i++ {
		for j := 0; j < numColumns/4; j++ {
			expected := A.Get(i+rowsStart, j+colsStart)
			actual := B.Get(i, j)
			if expected != actual {
				t.Log(fmt.Sprintf("wrong value in (%d,%d), expected %f, got %f", i, j, expected, actual))
				t.Fail()
			}
		}
	}

	if !t.Failed() {
		// Do again on the previous output
		numRows = 10
		numColumns = 10
		rowsStart := 0
		rowsEnd := numRows / 2
		colsStart := numColumns / 2
		colsEnd := numColumns
		C := B.GetMatrix(rowsStart, colsStart, rowsEnd, colsEnd)

		if C.Rows() != numRows/2 {
			t.Log(fmt.Sprintf("wrong number of rows, expected %d, got %d", numRows/2, C.Rows()))
			t.Fail()
		}
		if C.Cols() != numColumns/2 {
			t.Log(fmt.Sprintf("wrong number of columns, expected %d, got %d", numColumns/2, C.Cols()))
			t.Fail()
		}
		for i := 0; i < numRows/2; i++ {
			for j := 0; j < numColumns/2; j++ {
				expected := B.Get(i+rowsStart, j+colsStart)
				actual := C.Get(i, j)
				if expected != actual {
					t.Log(fmt.Sprintf("wrong value in (%d,%d), expected %f, got %f", i, j, expected, actual))
					t.Fail()
				}
			}
		}
	}
}

func TestAugment_Sparse(t *testing.T) {
	var A, B, C *SparseMatrix
	A = NormalsSparse(4, 4, 16)
	B = NormalsSparse(4, 4, 16)
	C, _ = A.Augment(B)
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < A.Cols(); j++ {
			if C.Get(i, j) != A.Get(i, j) {
				t.Fail()
			}
		}
	}
	for i := 0; i < B.Rows(); i++ {
		for j := 0; j < B.Cols(); j++ {
			if C.Get(i, j+A.Cols()) != B.Get(i, j) {
				t.Fail()
			}
		}
	}

	A = NormalsSparse(2, 2, 4)
	B = NormalsSparse(4, 4, 16)
	C, err := A.Augment(B)
	if err == nil {
		t.Fail()
	}

	A = NormalsSparse(4, 4, 16)
	B = NormalsSparse(4, 2, 8)
	C, _ = A.Augment(B)
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < A.Cols(); j++ {
			if C.Get(i, j) != A.Get(i, j) {
				t.Fail()
			}
		}
	}
	for i := 0; i < B.Rows(); i++ {
		for j := 0; j < B.Cols(); j++ {
			if C.Get(i, j+A.Cols()) != B.Get(i, j) {
				t.Fail()
			}
		}
	}
}

func TestStack_Sparse(t *testing.T) {
	var A, B, C *SparseMatrix
	A = NormalsSparse(4, 4, 16)
	B = NormalsSparse(4, 4, 16)
	C, _ = A.Stack(B)
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < A.Cols(); j++ {
			if C.Get(i, j) != A.Get(i, j) {
				t.Fail()
			}
		}
	}
	for i := 0; i < B.Rows(); i++ {
		for j := 0; j < B.Cols(); j++ {
			if C.Get(i+A.Rows(), j) != B.Get(i, j) {
				t.Fail()
			}
		}
	}

	A = NormalsSparse(2, 2, 4)
	B = NormalsSparse(4, 4, 16)
	C, err := A.Stack(B)
	if err == nil {
		if verbose {
			fmt.Printf("%v\n", err)
		}
		t.Fail()
	}

	A = NormalsSparse(4, 4, 16)
	B = NormalsSparse(2, 4, 8)
	C, _ = A.Stack(B)
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < A.Cols(); j++ {
			if C.Get(i, j) != A.Get(i, j) {
				t.Fail()
			}
		}
	}
	for i := 0; i < B.Rows(); i++ {
		for j := 0; j < B.Cols(); j++ {
			if C.Get(i+A.Rows(), j) != B.Get(i, j) {
				t.Fail()
			}
		}
	}
}

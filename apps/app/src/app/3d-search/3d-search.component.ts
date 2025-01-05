import {
  Component,
  inject,
  Inject,
  OnInit,
  Input,
  ChangeDetectionStrategy,
} from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import {
  AbstractControl,
  FormBuilder,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
  Validators,
  FormControl,
  FormGroupDirective,
  NgForm,
} from '@angular/forms';
import { ErrorStateMatcher } from '@angular/material/core';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatStepperModule } from '@angular/material/stepper';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { AuthService } from '../auth.service';
import { ImageSearchService } from '../image-search.service';
import { PlotlyService } from '../plotly.service';
import { Router } from '@angular/router';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { MatCardModule } from '@angular/material/card';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBar } from '@angular/material/snack-bar';
import {
  MatDialog,
  MatDialogActions,
  MatDialogClose,
  MatDialogContent,
  MatDialogRef,
  MatDialogTitle,
  MAT_DIALOG_DATA,
} from '@angular/material/dialog';

interface result {
  title: string;
  image: SafeUrl;
  similarity: number;
}

interface weights {
  dominant_colors: number;
  color_histogram: number;
  fourier_descriptors: number;
  hu_moments: number;
  edge_histogram: number;
  gabor: number;
}

interface advancedResults {
  images: result[];
  weights: weights;
  query_id: string;
}

interface imageDescriptors {
  dominant_colors: number[][];
  color_histogram: number[][];
  hu_moments: number[];
}

/** Error when invalid control is dirty, touched, or submitted. */
export class MyErrorStateMatcher implements ErrorStateMatcher {
  isErrorState(
    control: FormControl | null,
    form: FormGroupDirective | NgForm | null
  ): boolean {
    const isSubmitted = form && form.submitted;
    return !!(
      control &&
      control.invalid &&
      (control.dirty || control.touched || isSubmitted)
    );
  }
}

@Component({
  selector: 'app-3d-search',
  imports: [
    MatIconModule,
    MatTabsModule,
    MatButtonModule,
    MatStepperModule,
    FormsModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatCardModule,
    MatSelectModule,
  ],
  templateUrl: './3d-search.component.html',
  styleUrl: './3d-search.component.css',
})
export class ThreeDSearchComponent implements OnInit {
  private _formBuilder = inject(FormBuilder);
  private _authService = inject(AuthService);
  private _router = inject(Router);
  private _domSanitizer = inject(DomSanitizer);
  private _snackBar = inject(MatSnackBar);
  readonly _dialog = inject(MatDialog);

  openSnackBar(message: string, action: string) {
    return this._snackBar.open(message, action, {
      duration: 2000,
    });
  }

  selectedImageIndex: number | null = null;

  firstFormGroup = this._formBuilder.group({
    firstCtrl: ['', Validators.required],
    secondCtrl: ['', Validators.required],
  });

  secondFormGroup = this._formBuilder.group({
    secondCtrl: [null as number | null, Validators.required],
  });

  thirdCtrl = new FormControl('', Validators.required);
  thirdFormGroup = this._formBuilder.group({
    thirdCtrl: this.thirdCtrl,
  });

  matcher = new MyErrorStateMatcher();
  isLinear = true;
  uploadedFiles: { blob: File; sanitized: string }[] = [];
  objFileName = 'Select your 3D object';
  fileThumbnailName = "Select the object's thumbnail";
  results: result[] = [];

  ngOnInit(): void {
    if (!this._authService.isLoggedIn()) {
      this._router.navigate(['/login']);
    }
  }

  sanitize(url: string): SafeUrl {
    return this._domSanitizer.bypassSecurityTrustUrl(url);
  }

  updateObjFileName(event: any): void {
    this.objFileName = event.target.files[0].name;
  }

  updateThumbnailFileName(event: any): void {
    this.fileThumbnailName = event.target.files[0].name;
  }

  submitFile(): void {
    const objFileInput = document.getElementById(
      'ObjFileInput'
    ) as HTMLInputElement;
    const thumbnailInput = document.getElementById(
      'fileThumbnailInput'
    ) as HTMLInputElement;

    if (
      this.objFileName.replace(/\.[^/.]+$/, '') !==
      this.fileThumbnailName.replace(/\.[^/.]+$/, '')
    ) {
      let snackBarRef = this.openSnackBar(
        'Please select the same file for both fields',
        'Close'
      );

      snackBarRef.afterDismissed().subscribe(() => {
        return;
      });
      snackBarRef.onAction().subscribe(() => {
        return;
      });

      return;
    }

    if (objFileInput?.files && thumbnailInput?.files) {
      this.uploadedFiles.push({
        blob: objFileInput.files[0],
        sanitized: '',
      });

      const reader = new FileReader();
      reader.readAsDataURL(thumbnailInput.files[0]);

      reader.onload = (e) => {
        this.uploadedFiles[this.uploadedFiles.length - 1].sanitized = e.target
          ?.result as string;
      };

      this.objFileName = 'Select your 3D object';
      this.fileThumbnailName = "Select the object's thumbnail";
      objFileInput.value = '';
      thumbnailInput.value = '';
    }
  }

  selectCard(index: number): void {
    if (this.selectedImageIndex === index) {
      this.selectedImageIndex = null;
      this.secondFormGroup.get('secondCtrl')?.setValue(null);
    } else {
      this.selectedImageIndex = index;
      this.secondFormGroup.get('secondCtrl')?.setValue(index);
    }
  }

  deleteFile(index: number): void {
    if (this.selectedImageIndex === index) {
      this.selectedImageIndex = null;
      this.secondFormGroup.get('secondCtrl')?.setValue(null);
    }
    this.uploadedFiles.splice(index, 1);
  }

  deleteAllFiles(): void {
    const objFileInput = document.getElementById(
      'ObjFileInput'
    ) as HTMLInputElement;
    const thumbnailInput = document.getElementById(
      'fileThumbnailInput'
    ) as HTMLInputElement;

    this.uploadedFiles = [];
    this.selectedImageIndex = null;
    this.secondFormGroup.get('secondCtrl')?.setValue(null);
    this.results = [];
    this.objFileName = 'Select your 3D object';
    this.fileThumbnailName = "Select the object's thumbnail";
    objFileInput.value = '';
    thumbnailInput.value = '';
  }

  descriptors(index: number): void {
    this._dialog.open(DescriptorsDialog, {
      data: { index: index, uploadedFiles: this.uploadedFiles },
    });
  }

  logout(): void {
    this._authService.logout();
    this._router.navigate(['/login']);
  }
}

@Component({
  selector: 'descriptors-dialog',
  templateUrl: 'descriptors-dialog.html',
  imports: [
    MatButtonModule,
    MatDialogActions,
    MatDialogClose,
    MatDialogTitle,
    MatDialogContent,
  ],
  styleUrl: './3d-search.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DescriptorsDialog {
  constructor(
    private plot: PlotlyService,
    @Inject(MAT_DIALOG_DATA) public data: { index: number; uploadedFiles: any }
  ) {}

  private _imageSearchService = inject(ImageSearchService);
  readonly dialogRef = inject(MatDialogRef<DescriptorsDialog>);

  ngOnInit(): void {
    const { index, uploadedFiles } = this.data;
    const selectedFile = uploadedFiles[index].blob;
    let humoments: number[] = [];

    this._imageSearchService.imageDescriptors(selectedFile).subscribe(
      (results) => {
        console.log('Descriptors:', results);
        this.plot.plotHist(
          'histPlot',
          results.color_histogram[0],
          results.color_histogram[1],
          results.color_histogram[2]
        );

        this.plot.plotDominantColors(
          'dominantColorContainer',
          results.dominant_colors
        );

        humoments = results.hu_moments;
        const humomentsDiv = document.getElementById('humomentsContainer');
        for (let i = 0; i < humoments.length; i++) {
          const p = document.createElement('p');
          p.textContent = `Hu Moment ${i + 1}: ${humoments[i]}`;
          humomentsDiv?.appendChild(p);
        }
      },
      (error) => {
        console.error('Descriptors error:', error);
      }
    );
  }
}

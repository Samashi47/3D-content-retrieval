import {
  Component,
  inject,
  Inject,
  OnInit,
  Input,
  ChangeDetectionStrategy,
  ViewChild,
  ElementRef,
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
import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SearchService } from '../search.service';

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
  private _search = inject(SearchService);
  private _router = inject(Router);
  private _domSanitizer = inject(DomSanitizer);
  private _snackBar = inject(MatSnackBar);
  readonly _dialog = inject(MatDialog);

  openSnackBar(message: string, action: string) {
    return this._snackBar.open(message, action, {
      duration: 2000,
    });
  }

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
  selectedImageIndex: number | null = null;
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

  saveAs(blob: Blob, fileName: string) {
    const link = document.createElement('a');
    link.download = fileName;
    link.href = window.URL.createObjectURL(blob);
    link.click();
    window.URL.revokeObjectURL(link.href);
  }

  downloadModel(index: number): void {
    //const filename = this.results[index].title;
    const filename = 'model.obj';
    this._search.downloadModel(filename).subscribe(
      (response: any) => {
        console.log(response);
        this.saveAs(response, filename);
      },
      (error: any) => {
        console.log(error);
      }
    );
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
    @Inject(MAT_DIALOG_DATA) public data: { index: number; uploadedFiles: any }
  ) {}
  private canvas!: HTMLCanvasElement;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  readonly dialogRef = inject(MatDialogRef<DescriptorsDialog>);

  ngOnInit(): void {
    const { index, uploadedFiles } = this.data;
    const selectedFile = uploadedFiles[index].blob;
    const reader = new FileReader();
    this.initThree();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      const loader = new OBJLoader();
      const obj = loader.parse(content);
      obj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
          });
        }
      });
      obj.position.set(0, 0, 0);
      this.scene.add(obj);
    };
    reader.readAsText(selectedFile);
  }

  private initThree() {
    this.canvas = document.getElementById('canvas-box') as HTMLCanvasElement;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const skyColor = 0xb1e1ff;
    const groundColor = 0xb97a20;
    const intensity = 2;
    const hem_light = new THREE.HemisphereLight(
      skyColor,
      groundColor,
      intensity
    );
    this.scene.add(hem_light);
    const color = 0xffffff;

    const dir_light = new THREE.DirectionalLight(color, intensity);
    dir_light.position.set(-1000, 200, 200);
    dir_light.target.position.set(0, 0, 0);
    this.scene.add(dir_light);
    this.scene.add(dir_light.target);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
    });
    this.renderer.setSize(500, 500);
    this.camera.position.set(-1000, -200, -200);

    const controls = new OrbitControls(this.camera, this.canvas);
    controls.target.set(0, 0, 0);
    controls.update();

    this.animate();
  }

  private animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }
}

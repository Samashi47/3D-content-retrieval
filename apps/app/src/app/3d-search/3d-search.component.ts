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

interface searchResult {
  model_name: string;
  category: string;
  thumbnail: string;
  similarity: number;
}

interface descriptors {
  zernike: number[];
  fourier: number[];
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
  });

  secondFormGroup = this._formBuilder.group({
    secondCtrl: [null as number | null, Validators.required],
    thirdCtrl: ['5', Validators.required],
  });

  thirdCtrl = new FormControl('', Validators.required);
  thirdFormGroup = this._formBuilder.group({
    thirdCtrl: this.thirdCtrl,
  });

  matcher = new MyErrorStateMatcher();
  isLinear = true;
  uploadedFiles: { blob: File; sanitized: string }[] = [];
  selectedImageIndex: number | null = null;
  numberOfResults = '5';
  results: searchResult[] = [];

  ngOnInit(): void {
    if (!this._authService.isLoggedIn()) {
      this._router.navigate(['/login']);
    }
  }

  sanitize(url: string): SafeUrl {
    return this._domSanitizer.bypassSecurityTrustUrl(url);
  }

  selectFiles(event: any): void {
    if (event.target.files) {
      const l = this.uploadedFiles.length;
      for (let i = 0; i < event.target.files.length; i++) {
        const file = event.target.files[i];
        this.uploadedFiles.push({
          blob: file,
          sanitized: '',
        });
      }
    }
  }

  submitFile(): void {
    const objFileInput = document.getElementById(
      'ObjFileInput'
    ) as HTMLInputElement;

    if (objFileInput?.files) {
      this.uploadedFiles.push({
        blob: objFileInput.files[0],
        sanitized: '',
      });
      /*
      const reader = new FileReader();
      reader.readAsDataURL(thumbnailInput.files[0]);

      reader.onload = (e) => {
        this.uploadedFiles[this.uploadedFiles.length - 1].sanitized = e.target
          ?.result as string;
      };*/
      objFileInput.value = '';
    }
  }

  triggerThumbnailInput(index: any): void {
    const thumbnailInput = document.getElementById(
      'ThumbnailFileInput ' + index
    ) as HTMLInputElement;
    thumbnailInput.click();
  }

  submitThumbnail(index: any, event: any): void {
    console.log(index);
    if (event.target.files) {
      const file = event.target.files[0];

      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = (e) => {
        this.uploadedFiles[index].sanitized = e.target?.result as string;
      };
      event.target.value = '';
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

  selectNumberOfResults(): void {
    console.log(this.numberOfResults);
    const numberOfResults = parseInt(this.numberOfResults);
    if (this.results.length > numberOfResults) {
      this.results = this.results.slice(0, numberOfResults);
    } else {
      this.search();
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
    objFileInput.value = '';
    thumbnailInput.value = '';
  }

  descriptors(type: 'query' | 'result', index: number): void {
    if (type === 'query') {
      console.log(this.uploadedFiles[index].blob);
      this._search.queryDescriptors(this.uploadedFiles[index].blob).subscribe(
        (response: descriptors) => {
          console.log(response);
          this._dialog.open(DescriptorsDialog, {
            data: {
              blob: this.uploadedFiles[index].blob,
              zernike: response.zernike,
              fourier: response.fourier,
            },
          });
        },
        (error: any) => {
          console.log(error);
        }
      );
    } else {
      const filename = this.results[index].model_name;
      const category = this.results[index].category;
      let blob: File | null = null;
      this._search.downloadModel(filename, category).subscribe(
        (response: any) => {
          blob = response;
        },
        (error: any) => {
          console.log(error);
        }
      );

      this._search.resultDescriptors(filename, category).subscribe(
        (response: descriptors) => {
          console.log(response);
          this._dialog.open(DescriptorsDialog, {
            data: {
              blob: blob,
              zernike: response.zernike,
              fourier: response.fourier,
            },
          });
        },
        (error: any) => {
          console.log(error);
        }
      );
    }
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
    const filename = this.results[index].model_name;
    const category = this.results[index].category;
    this._search.downloadModel(filename, category).subscribe(
      (response: any) => {
        console.log(response);
        this.saveAs(response, filename + '.obj');
      },
      (error: any) => {
        console.log(error);
      }
    );
  }

  search(): void {
    const numberOfResults = parseInt(this.numberOfResults);

    if (this.selectedImageIndex !== null) {
      this._search
        .search(
          this.uploadedFiles[this.selectedImageIndex].blob,
          numberOfResults
        )
        .subscribe(
          (response: searchResult[]) => {
            console.log(response);
            this.results = response;
          },
          (error: any) => {
            console.log(error);
          }
        );
    } else {
      console.error('No image selected');
    }
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
    @Inject(MAT_DIALOG_DATA)
    public data: { blob: File; zernike: number[]; fourier: number[] }
  ) {}
  private canvas!: HTMLCanvasElement;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  readonly dialogRef = inject(MatDialogRef<DescriptorsDialog>);

  ngOnInit(): void {
    const { blob, zernike, fourier } = this.data;
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
      console.log('in here');
    };
    reader.readAsText(blob);
    console.log('out there');
    const descriptorsContainer = document.getElementById(
      'descriptors-container'
    );
    const zernikeContainer = document.createElement('div');
    zernikeContainer.innerHTML = '<h3>Zernike moments</h3>';
    for (let i = 0; i < zernike.length; i++) {
      const zernikeElement = document.createElement('div');
      zernikeElement.innerHTML = `Zernike ${i}: ${zernike[i]}`;
      zernikeContainer.appendChild(zernikeElement);
    }
    descriptorsContainer?.appendChild(zernikeContainer);

    const fourierContainer = document.createElement('div');
    fourierContainer.innerHTML = '<h3>Fourier descriptors</h3>';
    for (let i = 0; i < fourier.length; i++) {
      const fourierElement = document.createElement('div');
      fourierElement.innerHTML = `Fourier ${i}: ${fourier[i]}`;
      fourierContainer.appendChild(fourierElement);
    }
    descriptorsContainer?.appendChild(fourierContainer);
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
